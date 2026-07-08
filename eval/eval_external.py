"""External baseline per-item eval on the same 20k tone segments (same seed/selection).

Models: charactr/vocos-mel-24khz (English pretrained Vocos), nvidia/bigvgan_v2_24khz_100band_256x.
Each model uses its own native mel front-end. Metrics identical to internal eval.
Run from /data_nvme/vocos_training.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, "/data_nvme/vocos_training")
SCRATCH = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRATCH / "bigvgan_repo"))

from evaluate_tone_vocos import load_segment
from evaluate_tone_vocos_set import extract_f0, compare_f0
from vocos_train.tone_classifier import load_tone_classifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tone-manifest", default="manifests/tone_eval_val_no_overlap.tsv")
    parser.add_argument("--tone-evaluator-ckpt", default="checkpoints/tone_classifier/eval.pt")
    parser.add_argument("--max-items", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    evaluator = load_tone_classifier(args.tone_evaluator_ckpt, device=device).eval()
    n_tones = int(evaluator.n_tones)
    sample_rate = 24000

    from vocos import Vocos

    vocos_pre = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device).eval()

    import bigvgan as bigvgan_mod
    from meldataset import get_mel_spectrogram

    bv = bigvgan_mod.BigVGAN.from_pretrained("nvidia/bigvgan_v2_24khz_100band_256x", use_cuda_kernel=False)
    bv.remove_weight_norm()
    bv = bv.to(device).eval()

    def synth_vocos_pre(x: torch.Tensor) -> torch.Tensor:
        return vocos_pre(x)

    def synth_bigvgan(x: torch.Tensor) -> torch.Tensor:
        mel = get_mel_spectrogram(x, bv.h).to(device)
        return bv(mel).squeeze(1)

    models = [("vocos_pretrained_en", synth_vocos_pre), ("bigvgan_v2", synth_bigvgan)]

    with open(args.tone_manifest, "r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.max_items]
    rows = [(idx, row) for idx, row in enumerate(rows) if idx % args.num_shards == args.shard_index]

    out = Path(args.output)
    with out.open("w", encoding="utf-8", newline="") as handle, torch.inference_mode():
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            ["item_idx", "audio", "start", "end", "tone", "model", "pred", "correct", "f0_rmse", "f0_corr", "vuv_error"]
        )
        for idx, row in tqdm(rows, desc=f"ext-shard{args.shard_index}", mininterval=30):
            real = load_segment(row["audio"], float(row["start"]), float(row["end"]), sample_rate)
            if real.numel() < 1024:
                continue
            label = int(row["tone"])
            if label < 0 or label >= n_tones:
                continue
            x = real.unsqueeze(0).to(device)
            real_np = real.numpy()
            real_f0, real_voiced = extract_f0(real_np, sample_rate)
            for name, synth in models:
                fake = synth(x).float().cpu().squeeze(0)
                n = min(real.numel(), fake.numel())
                fake = fake[:n].clamp(-1, 1)
                pred = int(evaluator(fake.unsqueeze(0).to(device)).argmax(dim=1).item())
                metrics = compare_f0(real_f0, real_voiced, fake.numpy(), sample_rate)
                writer.writerow(
                    [
                        idx,
                        row["audio"],
                        row["start"],
                        row["end"],
                        label,
                        name,
                        pred,
                        int(pred == label),
                        f"{metrics['f0_rmse']:.6f}",
                        f"{metrics['f0_corr']:.6f}",
                        f"{metrics['vuv_error']:.6f}",
                    ]
                )


if __name__ == "__main__":
    main()
