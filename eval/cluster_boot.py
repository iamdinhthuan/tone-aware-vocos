# -*- coding: utf-8 -*-
"""Voice-level cluster bootstrap for every claim-bearing comparison.

Resamples the 80 held-out voices with replacement (2,000 draws) and recomputes the
mean paired difference, so the reported p-values do not assume the ~20k segments are
independent. Rewrites the `cluster_boot` block of revision_stats.json.

The first eight entries reproduce the original round-1 values bit-for-bit; the rest
were added so that no claim in the paper rests on an uncorrected-for-clustering test.
Run order is load-bearing: it fixes the RNG stream.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SEED = 20260706
N_BOOT = 2000

# (name, metric, system_a, system_b) -- order fixes the RNG stream, do not reorder.
COMPARISONS = [
    ("f0_c_vs_base", "f0_rmse", "plus_c", "baseline"),
    ("f0_cb_vs_base", "f0_rmse", "plus_cb", "baseline"),
    ("f0_cba_vs_base", "f0_rmse", "plus_cba", "baseline"),
    ("vuv_cb_vs_base", "vuv_error", "plus_cb", "baseline"),
    ("acc_cb_vs_base", "correct", "plus_cb", "baseline"),
    ("acc_cb_vs_bigvgan", "correct", "plus_cb", "bigvgan_v2"),
    ("acc_cb_vs_vocosen", "correct", "plus_cb", "vocos_pretrained_en"),
    ("f0_nga_cb_vs_base", "f0_rmse", "plus_cb", "baseline"),  # nga only
    ("f0_cb_vs_vocosen", "f0_rmse", "plus_cb", "vocos_pretrained_en"),
    ("vuv_cb_vs_vocosen", "vuv_error", "plus_cb", "vocos_pretrained_en"),
    ("f0corr_cb_vs_vocosen", "f0_corr", "plus_cb", "vocos_pretrained_en"),
    ("f0_cb_vs_bigvgan", "f0_rmse", "plus_cb", "bigvgan_v2"),
    ("f0_base_vs_bigvgan", "f0_rmse", "baseline", "bigvgan_v2"),
    ("vuv_cb_vs_bigvgan", "vuv_error", "plus_cb", "bigvgan_v2"),
    # Adjacent ablation steps. These carry the component-A regression claim, so they
    # belong here even though they are the ones that do not survive the check.
    ("f0_cba_vs_cb_adj", "f0_rmse", "plus_cba", "plus_cb"),
    ("f0corr_cba_vs_cb_adj", "f0_corr", "plus_cba", "plus_cb"),
    ("f0_cb_vs_c_adj", "f0_rmse", "plus_cb", "plus_c"),
]


def load() -> pd.DataFrame:
    frames = [
        pd.read_csv(HERE / "segment_peritem_internal.tsv.gz", sep="\t", low_memory=False),
        pd.read_csv(HERE / "segment_peritem_external.tsv.gz", sep="\t", low_memory=False),
    ]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["item_idx"] != "item_idx"].copy()
    for column in ("f0_rmse", "f0_corr", "vuv_error", "correct", "tone"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["key"] = df["audio"] + "|" + df["start"].astype(str) + "|" + df["end"].astype(str)
    basename = df["audio"].str.rsplit("/", n=1).str[-1]
    df["voice"] = basename.str.split("_", n=1).str[1].str.rsplit("_", n=1).str[0]
    return df


def cluster_boot(piv, voice_of, rng, metric, a, b, keys=None):
    x, y = piv[metric][a], piv[metric][b]
    if keys is not None:
        x, y = x[x.index.isin(keys)], y[y.index.isin(keys)]
    mask = np.isfinite(x) & np.isfinite(y)
    d = x[mask] - y[mask]
    voices = voice_of.loc[d.index]
    groups = [d.values[voices.values == g] for g in np.unique(voices.values)]
    n_groups = len(groups)
    means = np.empty(N_BOOT)
    for i in range(N_BOOT):
        pick = rng.integers(0, n_groups, size=n_groups)
        means[i] = np.concatenate([groups[j] for j in pick]).mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    p_boot = float(2 * min((means >= 0).mean(), (means <= 0).mean()))
    # The two-sided statistic can only land on multiples of 2/N_BOOT, so that -- not
    # 1/N_BOOT -- is the smallest positive value it can express.
    return dict(
        mean=float(d.mean()), lo=float(lo), hi=float(hi),
        p_boot=max(p_boot, 2.0 / N_BOOT), G=n_groups, n=int(mask.sum()),
    )


def main() -> None:
    df = load()
    piv = {
        m: df.pivot_table(index="key", columns="model", values=m, aggfunc="first")
        for m in ("f0_rmse", "f0_corr", "vuv_error", "correct")
    }
    voice_of = df.drop_duplicates("key").set_index("key")["voice"]
    tone_of = df.drop_duplicates("key").set_index("key")["tone"]
    rng = np.random.default_rng(SEED)

    out = {}
    for name, metric, a, b in COMPARISONS:
        keys = tone_of[tone_of == 4].index if "nga" in name else None
        out[name] = cluster_boot(piv, voice_of, rng, metric, a, b, keys=keys)
        r = out[name]
        print(f"{name:22s} mean={r['mean']:+.6f} CI[{r['lo']:+.4f},{r['hi']:+.4f}] "
              f"p_boot={r['p_boot']:.4f} n={r['n']} G={r['G']}")

    path = HERE / "revision_stats.json"
    stats = json.loads(path.read_text())
    stats["cluster_boot"] = out
    path.write_text(json.dumps(stats, indent=1) + "\n")
    print(f"\nwrote {len(out)} cluster-bootstrap entries to {path}")


if __name__ == "__main__":
    main()
