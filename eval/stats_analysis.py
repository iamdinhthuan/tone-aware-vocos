"""Statistical analysis for the EAAI paper.

Inputs: segment_peritem_internal.tsv.gz (4 in-house systems),
        segment_peritem_external.tsv.gz (2 external checkpoints),
        utterance_metrics_merged.tsv (6 systems).
The per-item tables carry one repeated header row per concatenated shard; they are filtered out on load.
Outputs (JSON + markdown-ish TSVs in scratchpad/analysis/):
  - segment_summary.tsv: per-system mean +/- bootstrap 95% CI for each metric
  - segment_stats.tsv: paired Wilcoxon vs chosen references with Holm correction + rank-biserial
  - per_tone_f0.tsv, per_tone_acc.tsv
  - utterance_summary.tsv, utterance_stats.tsv
McNemar test for paired binary tone-correctness.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
# The released repository keeps the shipped tables in <repo>/results and the scripts in
# <repo>/eval; the paper's own tree keeps both side by side. Resolve whichever applies, so
# one copy of this file works in both and the two cannot drift apart.
SP = HERE.parent / "results" if (HERE.parent / "results").is_dir() else HERE
OUT = SP / "analysis"
OUT.mkdir(exist_ok=True)

BASE_SEED = 20260706
TONES = ["ngang", "sac", "huyen", "hoi", "nga", "nang"]


def rng_for(key: str) -> np.random.Generator:
    """A generator determined solely by `key`, so an interval never depends on how many
    other intervals were drawn before it. A shared module-level generator would make every
    CI a function of call order, and reordering or adding a system would silently move
    intervals that had already been published."""
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return np.random.default_rng([BASE_SEED, int.from_bytes(digest, "little")])


def bootstrap_ci(values: np.ndarray, key: str, n_boot: int = 2000) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    mean = float(values.mean())
    idx = rng_for(key).integers(0, len(values), size=(n_boot, len(values)))
    boots = values[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return mean, float(lo), float(hi)


def rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation from Wilcoxon signed ranks."""
    d = x - y
    d = d[np.isfinite(d) & (d != 0)]
    if len(d) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    r_plus = ranks[d > 0].sum()
    r_minus = ranks[d < 0].sum()
    return float((r_plus - r_minus) / (r_plus + r_minus))


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


def mcnemar(a: np.ndarray, b: np.ndarray) -> float:
    """Exact-ish McNemar via binomial test on discordant pairs."""
    n01 = int(((a == 0) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    n = n01 + n10
    if n == 0:
        return 1.0
    return float(stats.binomtest(min(n01, n10), n, 0.5).pvalue * 1.0)


def load_segments() -> pd.DataFrame:
    frames = []
    for pattern in ["segment_peritem_internal.tsv.gz", "segment_peritem_external.tsv.gz"]:
        for f in sorted(SP.glob(pattern)):
            frames.append(pd.read_csv(f, sep="\t", low_memory=False))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["item_idx"] != "item_idx"].copy()  # drop the per-shard header rows
    for column in ("f0_rmse", "f0_corr", "vuv_error", "correct", "tone"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    key = df["audio"] + "|" + df["start"].astype(str) + "|" + df["end"].astype(str)
    df["key"] = key
    return df


def main() -> None:
    df = load_segments()
    systems = ["vocos_pretrained_en", "bigvgan_v2", "baseline", "plus_c", "plus_cb", "plus_cba"]
    metrics = ["correct", "f0_rmse", "f0_corr", "vuv_error"]

    # keep only keys present for ALL systems (paired analysis)
    counts = df.groupby("key")["model"].nunique()
    common = set(counts[counts == df["model"].nunique()].index)
    df = df[df["key"].isin(common)]
    print("systems found:", sorted(df["model"].unique()), "paired items:", len(common))

    wide = {m: df.pivot_table(index="key", columns="model", values=m, aggfunc="first") for m in metrics}
    tone_of_key = df.drop_duplicates("key").set_index("key")["tone"]

    # --- summary with bootstrap CI ---
    rows = []
    for s in systems:
        if s not in wide["correct"].columns:
            continue
        row = {"model": s, "n": int(wide["correct"][s].notna().sum())}
        for m in metrics:
            mean, lo, hi = bootstrap_ci(wide[m][s].to_numpy(dtype=float), key=f"segment|{s}|{m}")
            row[m] = mean
            row[f"{m}_lo"], row[f"{m}_hi"] = lo, hi
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "segment_summary.tsv", sep="\t", index=False)

    # --- paired tests ---
    comparisons = []
    for target in ["plus_c", "plus_cb", "plus_cba"]:
        comparisons.append(("baseline", target))
    for ext in ["vocos_pretrained_en", "bigvgan_v2"]:
        comparisons.append((ext, "plus_cb"))
        comparisons.append((ext, "baseline"))

    stat_rows = []
    for metric in ["f0_rmse", "f0_corr", "vuv_error"]:
        pvals, entries = [], []
        for ref, target in comparisons:
            if ref not in wide[metric].columns or target not in wide[metric].columns:
                continue
            x = wide[metric][target].to_numpy(dtype=float)
            y = wide[metric][ref].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            res = stats.wilcoxon(x[mask], y[mask], zero_method="wilcox", method="approx")
            eff = rank_biserial(x[mask], y[mask])
            pvals.append(float(res.pvalue))
            entries.append(
                {
                    "metric": metric,
                    "comparison": f"{target} vs {ref}",
                    "n": int(mask.sum()),
                    "mean_target": float(np.nanmean(x[mask])),
                    "mean_ref": float(np.nanmean(y[mask])),
                    "delta": float(np.nanmean(x[mask] - y[mask])),
                    "p_raw": float(res.pvalue),
                    "effect_rank_biserial": eff,
                }
            )
        for entry, p_adj in zip(entries, holm(pvals)):
            entry["p_holm"] = p_adj
            stat_rows.append(entry)

    # tone correctness: McNemar
    pvals, entries = [], []
    for ref, target in comparisons:
        if ref not in wide["correct"].columns or target not in wide["correct"].columns:
            continue
        x = wide["correct"][target].to_numpy(dtype=float)
        y = wide["correct"][ref].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        p = mcnemar(y[mask].astype(int), x[mask].astype(int))
        pvals.append(p)
        entries.append(
            {
                "metric": "tone_accuracy",
                "comparison": f"{target} vs {ref}",
                "n": int(mask.sum()),
                "mean_target": float(x[mask].mean()),
                "mean_ref": float(y[mask].mean()),
                "delta": float(x[mask].mean() - y[mask].mean()),
                "p_raw": p,
                "effect_rank_biserial": float("nan"),
            }
        )
    for entry, p_adj in zip(entries, holm(pvals)):
        entry["p_holm"] = p_adj
        stat_rows.append(entry)

    pd.DataFrame(stat_rows).to_csv(OUT / "segment_stats.tsv", sep="\t", index=False)

    # --- per-tone breakdown ---
    acc_rows, f0_rows = [], []
    for s in systems:
        if s not in wide["correct"].columns:
            continue
        arow, frow = {"model": s}, {"model": s}
        for t_id, t_name in enumerate(TONES):
            keys = tone_of_key[tone_of_key == t_id].index
            arow[t_name] = float(wide["correct"][s].loc[wide["correct"].index.isin(keys)].mean())
            frow[t_name] = float(np.nanmean(wide["f0_rmse"][s].loc[wide["f0_rmse"].index.isin(keys)]))
        acc_rows.append(arow)
        f0_rows.append(frow)
    pd.DataFrame(acc_rows).to_csv(OUT / "per_tone_acc.tsv", sep="\t", index=False)
    pd.DataFrame(f0_rows).to_csv(OUT / "per_tone_f0.tsv", sep="\t", index=False)

    # --- utterance level ---
    up = SP / "utterance_metrics_merged.tsv"
    if up.exists():
        u = pd.read_csv(up, sep="\t")
        um = ["utmos", "pesq_wb", "estoi", "mcd_db"]
        uw = {m: u.pivot_table(index="file", columns="model", values=m, aggfunc="first") for m in um}
        urows = []
        ref_utmos = u.drop_duplicates("file")["utmos_ref"].astype(float)
        urows.append({"model": "ground_truth", "n": len(ref_utmos), "utmos": float(ref_utmos.mean())})
        for s in systems:
            if s not in uw["utmos"].columns:
                continue
            row = {"model": s, "n": int(uw["utmos"][s].notna().sum())}
            for m in um:
                mean, lo, hi = bootstrap_ci(uw[m][s].to_numpy(dtype=float), key=f"utterance|{s}|{m}")
                row[m], row[f"{m}_lo"], row[f"{m}_hi"] = mean, lo, hi
            urows.append(row)
        pd.DataFrame(urows).to_csv(OUT / "utterance_summary.tsv", sep="\t", index=False)

        stat_rows = []
        for metric in um:
            pvals, entries = [], []
            for ref, target in comparisons:
                if ref not in uw[metric].columns or target not in uw[metric].columns:
                    continue
                x = uw[metric][target].to_numpy(dtype=float)
                y = uw[metric][ref].to_numpy(dtype=float)
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() < 10:
                    continue
                res = stats.wilcoxon(x[mask], y[mask], zero_method="wilcox", method="approx")
                pvals.append(float(res.pvalue))
                entries.append(
                    {
                        "metric": metric,
                        "comparison": f"{target} vs {ref}",
                        "n": int(mask.sum()),
                        "mean_target": float(np.nanmean(x[mask])),
                        "mean_ref": float(np.nanmean(y[mask])),
                        "delta": float(np.nanmean(x[mask] - y[mask])),
                        "p_raw": float(res.pvalue),
                        "effect_rank_biserial": rank_biserial(x[mask], y[mask]),
                    }
                )
            for entry, p_adj in zip(entries, holm(pvals)):
                entry["p_holm"] = p_adj
                stat_rows.append(entry)
        pd.DataFrame(stat_rows).to_csv(OUT / "utterance_stats.tsv", sep="\t", index=False)

    print("analysis written to", OUT)


if __name__ == "__main__":
    main()
