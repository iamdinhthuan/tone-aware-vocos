from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate sharded evaluate_tone_vocos_set.py JSON reports")
    parser.add_argument("--input-dir", default="eval_reports")
    parser.add_argument("--prefix", default="tone_val_no_overlap")
    parser.add_argument("--max-items-label", default="20k")
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--output-dir", default="eval_reports")
    return parser.parse_args()


def load_shards(input_dir: Path, prefix: str, name: str, max_items_label: str, num_shards: int) -> list[dict]:
    shards = []
    for idx in range(num_shards):
        path = input_dir / f"{prefix}_{name}_{max_items_label}_shard{idx:02d}of{num_shards:02d}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        shards.append(json.loads(path.read_text(encoding="utf-8")))
    return shards


def aggregate_name(input_dir: Path, output_dir: Path, prefix: str, name: str, max_items_label: str, num_shards: int) -> tuple[str, dict, Path]:
    shards = load_shards(input_dir, prefix, name, max_items_label, num_shards)
    confusion = None
    total_items = 0
    metric_sums = {"f0_rmse": 0.0, "f0_corr": 0.0, "vuv_error": 0.0}
    metric_counts = {"f0_rmse": 0, "f0_corr": 0, "vuv_error": 0}
    for shard in shards:
        c = np.array(shard["confusion"], dtype=np.int64)
        confusion = c if confusion is None else confusion + c
        total_items += int(shard["items"])
        for key in metric_sums:
            metric_sums[key] += float(shard.get("metric_sums", {}).get(key, shard[key] * shard["items"]))
            metric_counts[key] += int(shard.get("metric_counts", {}).get(key, shard["items"]))
    assert confusion is not None
    correct = int(np.trace(confusion))
    def mean(key: str) -> float:
        count = metric_counts[key]
        return float(metric_sums[key] / count) if count else float("nan")

    report = {
        "checkpoint": shards[0]["checkpoint"],
        "tone_manifest": shards[0]["tone_manifest"],
        "manifest_items": shards[0]["manifest_items"],
        "max_items": shards[0]["max_items"],
        "selected_items": shards[0].get("selected_items", shards[0]["max_items"]),
        "num_shards": num_shards,
        "shuffle": shards[0]["shuffle"],
        "seed": shards[0]["seed"],
        "items": total_items,
        "tone_accuracy": correct / max(1, total_items),
        "confusion": confusion.tolist(),
        "f0_rmse": mean("f0_rmse"),
        "f0_corr": mean("f0_corr"),
        "vuv_error": mean("vuv_error"),
        "metric_sums": metric_sums,
        "metric_counts": metric_counts,
        "shards": [f"{prefix}_{name}_{max_items_label}_shard{idx:02d}of{num_shards:02d}.json" for idx in range(num_shards)],
    }
    output = output_dir / f"{prefix}_{name}_{max_items_label}.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return name, report, output


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = []
    pattern = input_dir / f"{args.prefix}_*_{args.max_items_label}_shard00of{args.num_shards:02d}.json"
    for path in sorted(glob.glob(str(pattern))):
        stem = Path(path).name
        left = f"{args.prefix}_"
        right = f"_{args.max_items_label}_shard00of{args.num_shards:02d}.json"
        names.append(stem[len(left) : -len(right)])
    if not names:
        raise FileNotFoundError(f"No shard00 files match {pattern}")

    summary_rows = []
    for name in names:
        model_name, report, output = aggregate_name(input_dir, output_dir, args.prefix, name, args.max_items_label, args.num_shards)
        summary_rows.append((model_name, report, output))
        print(json.dumps({"name": model_name, "output": str(output), **report}, indent=2, ensure_ascii=False))

    summary_path = output_dir / f"{args.prefix}_summary_{args.max_items_label}.tsv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
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
