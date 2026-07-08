from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from vocos_train.model import VocosGenerator
from vocos_train.tone_classifier import load_tone_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Vietnamese tone-aware Vocos checkpoints")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tone-manifest", required=True, help="TSV from extract_tone_segments.py")
    parser.add_argument("--tone-evaluator-ckpt", required=True, help="Independent tone classifier checkpoint")
    parser.add_argument("--output", default="eval_reports/tone_vocos.json")
    parser.add_argument("--max-items", type=int, default=2000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_segment(path: str, start: float, end: float, sample_rate: int) -> torch.Tensor:
    info = torchaudio.info(path)
    frame_offset = max(0, int(start * info.sample_rate))
    num_frames = max(1, int((end - start) * info.sample_rate))
    wav, sr = torchaudio.load(path, frame_offset=frame_offset, num_frames=num_frames)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav[0].float()


def f0_metrics(real: np.ndarray, fake: np.ndarray, sample_rate: int) -> dict[str, float]:
    import librosa

    frame_length = 1024
    hop_length = 256
    f0_real, voiced_real, _ = librosa.pyin(
        real, fmin=60, fmax=500, sr=sample_rate, frame_length=frame_length, hop_length=hop_length
    )
    f0_fake, voiced_fake, _ = librosa.pyin(
        fake, fmin=60, fmax=500, sr=sample_rate, frame_length=frame_length, hop_length=hop_length
    )
    n = min(len(f0_real), len(f0_fake), len(voiced_real), len(voiced_fake))
    f0_real, f0_fake = f0_real[:n], f0_fake[:n]
    voiced_real, voiced_fake = voiced_real[:n], voiced_fake[:n]
    mask = voiced_real & voiced_fake & np.isfinite(f0_real) & np.isfinite(f0_fake)
    if mask.sum() < 2:
        rmse = corr = float("nan")
    else:
        err = f0_fake[mask] - f0_real[mask]
        rmse = float(np.sqrt(np.mean(err * err)))
        corr = float(np.corrcoef(f0_real[mask], f0_fake[mask])[0, 1])
    vuv = float(np.mean(voiced_real != voiced_fake)) if n else float("nan")
    return {"f0_rmse": rmse, "f0_corr": corr, "vuv_error": vuv}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    sample_rate = config["data"]["sample_rate"]
    generator = VocosGenerator(config["data"], config["model"]).to(device).eval()
    generator.load_state_dict(checkpoint["generator"])
    evaluator = load_tone_classifier(args.tone_evaluator_ckpt, device=device).eval()

    rows = []
    with open(args.tone_manifest, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows.append(row)
            if len(rows) >= args.max_items:
                break

    confusion = np.zeros((6, 6), dtype=np.int64)
    metrics = []
    correct = total = 0
    with torch.inference_mode():
        for row in tqdm(rows, desc="eval"):
            real = load_segment(row["audio"], float(row["start"]), float(row["end"]), sample_rate)
            if real.numel() < 1024:
                continue
            x = real.unsqueeze(0).to(device)
            fake = generator(x).float().cpu().squeeze(0)
            n = min(real.numel(), fake.numel())
            real = real[:n]
            fake = fake[:n].clamp(-1, 1)
            label = int(row["tone"])
            pred = int(evaluator(fake.unsqueeze(0).to(device)).argmax(dim=1).item())
            confusion[label, pred] += 1
            correct += int(pred == label)
            total += 1
            f0 = f0_metrics(real.numpy(), fake.numpy(), sample_rate)
            metrics.append(f0)

    def nanmean(key: str) -> float:
        vals = np.array([item[key] for item in metrics], dtype=np.float64)
        return float(np.nanmean(vals)) if vals.size else float("nan")

    report = {
        "checkpoint": args.checkpoint,
        "items": total,
        "tone_accuracy": correct / max(1, total),
        "confusion": confusion.tolist(),
        "f0_rmse": nanmean("f0_rmse"),
        "f0_corr": nanmean("f0_corr"),
        "vuv_error": nanmean("vuv_error"),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
