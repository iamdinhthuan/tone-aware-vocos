"""Per-item re-run of evaluate_tone_vocos_set.py (same seed/selection) dumping one row per segment per model.

Run from /data_nvme/vocos_training with the py310 env.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, "/data_nvme/vocos_training")

from evaluate_tone_vocos import load_segment
from evaluate_tone_vocos_set import extract_f0, compare_f0, load_generator
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

    specs = [
        ("baseline", "checkpoints/vocos_mp3/best.pt"),
        ("plus_c", "checkpoints/ablations/plus_c/best.pt"),
        ("plus_cb", "checkpoints/ablations/plus_cb/best.pt"),
        ("plus_cba", "checkpoints/ablations/plus_cba/best.pt"),
    ]
    generators = []
    sample_rate = None
    for name, path in specs:
        gen, sr = load_generator(path, device)
        sample_rate = sample_rate or sr
        generators.append((name, gen))

    rows = []
    with open(args.tone_manifest, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.max_items]
    rows = [(idx, row) for idx, row in enumerate(rows) if idx % args.num_shards == args.shard_index]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle, torch.inference_mode():
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            ["item_idx", "audio", "start", "end", "tone", "model", "pred", "correct", "f0_rmse", "f0_corr", "vuv_error"]
        )
        for idx, row in tqdm(rows, desc=f"shard{args.shard_index}", mininterval=30):
            real = load_segment(row["audio"], float(row["start"]), float(row["end"]), sample_rate)
            if real.numel() < 1024:
                continue
            label = int(row["tone"])
            if label < 0 or label >= n_tones:
                continue
            x = real.unsqueeze(0).to(device)
            real_np = real.numpy()
            real_f0, real_voiced = extract_f0(real_np, sample_rate)
            for name, gen in generators:
                fake = gen(x).float().cpu().squeeze(0)
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
