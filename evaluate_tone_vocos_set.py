from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from evaluate_tone_vocos import load_segment
from vocos_train.model import VocosGenerator
from vocos_train.tone_classifier import load_tone_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a set of Vietnamese tone-aware Vocos checkpoints on the same held-out tone manifest"
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Named checkpoint in the form name=/path/to/best.pt. Can be repeated.",
    )
    parser.add_argument("--tone-manifest", required=True, help="TSV from extract_tone_segments.py")
    parser.add_argument("--tone-evaluator-ckpt", required=True, help="Independent tone classifier checkpoint")
    parser.add_argument("--output-dir", default="eval_reports")
    parser.add_argument("--output-prefix", default="tone_val")
    parser.add_argument("--max-items", type=int, default=20000)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle manifest rows before taking --max-items")
    parser.add_argument("--seed", type=int, default=20260706, help="Deterministic row-shuffle seed")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of evaluation shards")
    parser.add_argument("--shard-index", type=int, default=0, help="This shard index in [0, num_shards)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@dataclass
class ModelEvalState:
    name: str
    checkpoint_path: str
    generator: VocosGenerator
    confusion: np.ndarray
    correct: int = 0
    total: int = 0
    metrics: list[dict[str, float]] = field(default_factory=list)


def parse_checkpoint_specs(specs: list[str]) -> list[tuple[str, str]]:
    parsed = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --checkpoint spec {spec!r}; expected name=/path/to/checkpoint.pt")
        name, path = spec.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"Invalid --checkpoint spec {spec!r}; expected non-empty name and path")
        parsed.append((name, path))
    return parsed


def load_generator(path: str, device: torch.device) -> tuple[VocosGenerator, int]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    sample_rate = int(config["data"]["sample_rate"])
    generator = VocosGenerator(config["data"], config["model"]).to(device).eval()
    generator.load_state_dict(checkpoint["generator"])
    return generator, sample_rate


def extract_f0(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    import librosa

    f0, voiced, _ = librosa.pyin(
        audio,
        fmin=60,
        fmax=500,
        sr=sample_rate,
        frame_length=1024,
        hop_length=256,
    )
    return f0, voiced


def compare_f0(
    real_f0: np.ndarray,
    real_voiced: np.ndarray,
    fake_audio: np.ndarray,
    sample_rate: int,
) -> dict[str, float]:
    fake_f0, fake_voiced = extract_f0(fake_audio, sample_rate)
    n = min(len(real_f0), len(fake_f0), len(real_voiced), len(fake_voiced))
    real_f0 = real_f0[:n]
    fake_f0 = fake_f0[:n]
    real_voiced = real_voiced[:n]
    fake_voiced = fake_voiced[:n]
    mask = real_voiced & fake_voiced & np.isfinite(real_f0) & np.isfinite(fake_f0)
    if mask.sum() < 2:
        rmse = corr = float("nan")
    else:
        err = fake_f0[mask] - real_f0[mask]
        rmse = float(np.sqrt(np.mean(err * err)))
        corr = float(np.corrcoef(real_f0[mask], fake_f0[mask])[0, 1])
    vuv = float(np.mean(real_voiced != fake_voiced)) if n else float("nan")
    return {"f0_rmse": rmse, "f0_corr": corr, "vuv_error": vuv}


def nanmean(metrics: list[dict[str, float]], key: str) -> float:
    vals = np.array([item[key] for item in metrics], dtype=np.float64)
    return float(np.nanmean(vals)) if vals.size else float("nan")


def nansum_and_count(metrics: list[dict[str, float]], key: str) -> tuple[float, int]:
    vals = np.array([item[key] for item in metrics], dtype=np.float64)
    mask = np.isfinite(vals)
    return (float(vals[mask].sum()), int(mask.sum()))


def suffix_for_max_items(max_items: int) -> str:
    if max_items > 0 and max_items % 1000 == 0:
        return f"{max_items // 1000}k"
    return str(max_items)


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    device = torch.device(args.device)
    checkpoint_specs = parse_checkpoint_specs(args.checkpoint)
    evaluator = load_tone_classifier(args.tone_evaluator_ckpt, device=device).eval()
    n_tones = int(evaluator.n_tones)

    states: list[ModelEvalState] = []
    sample_rate: int | None = None
    for name, path in checkpoint_specs:
        generator, ckpt_sample_rate = load_generator(path, device)
        if sample_rate is None:
            sample_rate = ckpt_sample_rate
        elif ckpt_sample_rate != sample_rate:
            raise ValueError(
                f"Checkpoint {path} has sample_rate={ckpt_sample_rate}, expected {sample_rate}"
            )
        states.append(
            ModelEvalState(
                name=name,
                checkpoint_path=path,
                generator=generator,
                confusion=np.zeros((n_tones, n_tones), dtype=np.int64),
            )
        )
    assert sample_rate is not None

    rows = []
    with open(args.tone_manifest, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows.append(row)
    manifest_items = len(rows)
    if args.shuffle:
        random.Random(args.seed).shuffle(rows)
    if args.max_items > 0:
        rows = rows[: args.max_items]
    selected_items = len(rows)
    if args.num_shards > 1:
        rows = [row for idx, row in enumerate(rows) if idx % args.num_shards == args.shard_index]

    with torch.inference_mode():
        for row in tqdm(rows, desc="eval-set"):
            real = load_segment(row["audio"], float(row["start"]), float(row["end"]), sample_rate)
            if real.numel() < 1024:
                continue
            label = int(row["tone"])
            if label < 0 or label >= n_tones:
                continue
            x = real.unsqueeze(0).to(device)
            real_np = real.numpy()
            real_f0, real_voiced = extract_f0(real_np, sample_rate)

            for state in states:
                fake = state.generator(x).float().cpu().squeeze(0)
                n = min(real.numel(), fake.numel())
                fake = fake[:n].clamp(-1, 1)
                pred = int(evaluator(fake.unsqueeze(0).to(device)).argmax(dim=1).item())
                state.confusion[label, pred] += 1
                state.correct += int(pred == label)
                state.total += 1
                state.metrics.append(compare_f0(real_f0, real_voiced, fake.numpy(), sample_rate))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = suffix_for_max_items(args.max_items)
    if args.num_shards > 1:
        suffix = f"{suffix}_shard{args.shard_index:02d}of{args.num_shards:02d}"
    summary_rows = []
    for state in states:
        metric_sums = {}
        metric_counts = {}
        for key in ["f0_rmse", "f0_corr", "vuv_error"]:
            metric_sums[key], metric_counts[key] = nansum_and_count(state.metrics, key)
        report = {
            "checkpoint": state.checkpoint_path,
            "tone_manifest": args.tone_manifest,
            "manifest_items": manifest_items,
            "max_items": args.max_items,
            "selected_items": selected_items,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "shuffle": bool(args.shuffle),
            "seed": args.seed if args.shuffle else None,
            "items": state.total,
            "tone_accuracy": state.correct / max(1, state.total),
            "confusion": state.confusion.tolist(),
            "f0_rmse": nanmean(state.metrics, "f0_rmse"),
            "f0_corr": nanmean(state.metrics, "f0_corr"),
            "vuv_error": nanmean(state.metrics, "vuv_error"),
            "metric_sums": metric_sums,
            "metric_counts": metric_counts,
        }
        output = output_dir / f"{args.output_prefix}_{state.name}_{suffix}.json"
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        summary_rows.append((state.name, report, output))
        print(json.dumps({"name": state.name, "output": str(output), **report}, indent=2, ensure_ascii=False))

    summary_path = output_dir / f"{args.output_prefix}_summary_{suffix}.tsv"
    with summary_path.open("w", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["name", "items", "tone_accuracy", "f0_rmse", "f0_corr", "vuv_error", "checkpoint", "json"])
        for name, report, output in summary_rows:
            writer.writerow(
                [
                    name,
                    report["items"],
                    f"{report['tone_accuracy']:.10f}",
                    f"{report['f0_rmse']:.10f}",
                    f"{report['f0_corr']:.10f}",
                    f"{report['vuv_error']:.10f}",
                    report["checkpoint"],
                    output,
                ]
            )
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
