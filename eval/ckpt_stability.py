# -*- coding: utf-8 -*-
"""Within-run checkpoint stability (R1 round-2): F0 RMSE for 3 late checkpoints per run
on a fixed 5,000-segment subset of the standard 20k selection.
Run from /data_nvme/vocos_training.
"""
from __future__ import annotations
import argparse, csv, random, sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, "/data_nvme/vocos_training")
from evaluate_tone_vocos import load_segment
from evaluate_tone_vocos_set import extract_f0, compare_f0, load_generator

CKPTS = [
    ("baseline", "s965k", "checkpoints/vocos_mp3/best_step_965000_val_0.161656.pt"),
    ("baseline", "s990k", "checkpoints/vocos_mp3/best_step_990000_val_0.161787.pt"),
    ("baseline", "s1000k", "checkpoints/vocos_mp3/best_step_1000000_val_0.161822.pt"),
    ("plus_c", "s955k", "checkpoints/ablations/plus_c/best_step_955000_val_0.162642.pt"),
    ("plus_c", "s990k", "checkpoints/ablations/plus_c/best_step_990000_val_0.162639.pt"),
    ("plus_c", "s1000k", "checkpoints/ablations/plus_c/best_step_1000000_val_0.162648.pt"),
    ("plus_cb", "s960k", "checkpoints/ablations/plus_cb/best_step_960000_val_0.163419.pt"),
    ("plus_cb", "s995k", "checkpoints/ablations/plus_cb/best_step_995000_val_0.163416.pt"),
    ("plus_cb", "s1000k", "checkpoints/ablations/plus_cb/best_step_1000000_val_0.163414.pt"),
    ("plus_cba", "s990k", "checkpoints/ablations/plus_cba/best_step_990000_val_0.166101.pt"),
    ("plus_cba", "s995k", "checkpoints/ablations/plus_cba/best_step_995000_val_0.166087.pt"),
    ("plus_cba", "s1000k", "checkpoints/ablations/plus_cba/best_step_1000000_val_0.166086.pt"),
]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-shards", type=int, default=4)
    ap.add_argument("--shard-index", type=int, required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    device = torch.device("cuda")

    gens = []
    for run, tag, path in CKPTS:
        g, _ = load_generator(path, device)
        gens.append((f"{run}|{tag}", g))

    rows = list(csv.DictReader(open("manifests/tone_eval_val_no_overlap.tsv"), delimiter="\t"))
    random.Random(20260706).shuffle(rows)
    rows = rows[:5000]
    rows = [(i, r) for i, r in enumerate(rows) if i % args.num_shards == args.shard_index]

    out = Path(args.output)
    with out.open("w", newline="", encoding="utf-8") as h, torch.inference_mode():
        w = csv.writer(h, delimiter="\t")
        w.writerow(["item_idx", "model", "f0_rmse"])
        for idx, row in tqdm(rows, mininterval=30):
            real = load_segment(row["audio"], float(row["start"]), float(row["end"]), 24000)
            if real.numel() < 1024:
                continue
            x = real.unsqueeze(0).to(device)
            rf0, rv = extract_f0(real.numpy(), 24000)
            for name, g in gens:
                fake = g(x).float().cpu().squeeze(0)
                n = min(real.numel(), fake.numel())
                m = compare_f0(rf0, rv, fake[:n].clamp(-1, 1).numpy(), 24000)
                w.writerow([idx, name, f"{m['f0_rmse']:.6f}"])

if __name__ == "__main__":
    main()
