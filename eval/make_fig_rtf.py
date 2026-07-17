# -*- coding: utf-8 -*-
"""Fig 8: real-time factor vs tone accuracy.

Tone accuracies are READ FROM segment_summary.tsv, never hard-coded: they were hard-coded
once and drifted from Table 3 the moment the table was regenerated.

All four matched-budget systems share one x by construction -- the tone-aware terms are
training-time only, so inference cost is identical. Baseline and +C additionally sit within
0.005 pp of each other, i.e. they genuinely coincide; the baseline marker is therefore drawn
larger and beneath +C so it reads as a ring rather than vanishing under it.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
SP = HERE.parent / "results" if (HERE.parent / "results").is_dir() else HERE
TABLES = SP / "analysis" if (SP / "analysis" / "segment_summary.tsv").exists() else SP
FIG = HERE.parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e6", "axes.axisbelow": True,
                     "pdf.fonttype": 42})

eff = {r["model"]: r for r in json.load(open(SP / "efficiency.json"))}
real_acc = json.load(open(SP / "real_acc.json"))["real_tone_accuracy"] * 100
acc = pd.read_csv(TABLES / "segment_summary.tsv", sep="\t").set_index("model")["correct"] * 100
RTF_OURS = eff["ours_vocos_vi"]["rtf_gpu"]

#        label          x                                    y                        colour     marker  size
points = [
    ("Vocos-EN",   eff["vocos_pretrained_en"]["rtf_gpu"], acc["vocos_pretrained_en"], "#E69F00", "s", 52),
    ("BigVGAN-v2", eff["bigvgan_v2"]["rtf_gpu"],          acc["bigvgan_v2"],          "#009E73", "D", 52),
    # baseline first and largest: +C lands on top of it and would otherwise hide it entirely
    ("Baseline",   RTF_OURS,                              acc["baseline"],            "#0072B2", "o", 96),
    ("+C",         RTF_OURS,                              acc["plus_c"],              "#56B4E9", "o", 34),
    ("+C+B",       RTF_OURS,                              acc["plus_cb"],             "#D55E00", "o", 46),
    ("+C+B+A",     RTF_OURS,                              acc["plus_cba"],            "#CC79A7", "o", 46),
]

fig, ax = plt.subplots(figsize=(5.4, 2.9))
for name, x, y, c, m, s in points:
    ax.scatter(x, y, s=s, color=c, marker=m, zorder=3, edgecolor="white", linewidth=0.6)

# The four matched-budget systems share one x, and baseline/+C differ by 0.005 pp, so plain
# offset labels collide and detach from their markers. Give that column leader lines to a
# staggered label column instead; the two isolated externals keep simple direct labels.
LEADER_X = RTF_OURS * 3.0
leader_y = {"+C+B": 57.98, "Baseline": 57.60, "+C": 57.28, "+C+B+A": 56.98}
for name, x, y, c, m, s_ in points:
    if name in leader_y:
        ly = leader_y[name]
        ax.annotate("", xy=(x * 1.25, y), xytext=(LEADER_X * 0.94, ly),
                    arrowprops=dict(arrowstyle="-", color="#9a9a9a", lw=0.6,
                                    shrinkA=1, shrinkB=1))
        ax.text(LEADER_X, ly, name, fontsize=7.5, ha="left", va="center", color="#1a1a1a",
                fontweight="bold" if name == "+C+B" else "normal")
    else:
        dx, dy = {"Vocos-EN": (0, -12), "BigVGAN-v2": (0, 9)}[name]
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(dx, dy),
                    fontsize=7.5, ha="center", va="center", color="#1a1a1a")

ax.axhline(real_acc, color="black", ls="--", lw=1.0)
ax.text(1.5e-4, real_acc + 0.08, "source recordings (ceiling)", fontsize=7.5)
ax.set_xscale("log")
ax.set_xlabel("Real-time factor on RTX 4090 (log scale, lower = faster)")
ax.set_ylabel("Tone accuracy (%)")
ax.set_xlim(1.2e-4, 3e-2)
ax.set_ylim(56.7, 59.3)
ax.annotate("", xy=(RTF_OURS * 1.5, acc["plus_cb"]), xytext=(RTF_OURS * 1.5, acc["baseline"]),
            arrowprops=dict(arrowstyle="->", color="#D55E00", lw=1.2))
ax.text(RTF_OURS * 11, 57.62, "same cost,\nbetter tone", fontsize=7.5, color="#D55E00", va="center")
fig.tight_layout()
fig.savefig(FIG / "fig_rtf_quality.pdf", bbox_inches="tight", dpi=300)
fig.savefig(FIG / "fig_rtf_quality.png", dpi=300, bbox_inches="tight")

# --- self-check: no label may start left of the axes, and every system must be drawn ---
import fitz  # noqa: E402

d = fitz.open(FIG / "fig_rtf_quality.pdf")
x0_axes = ax.get_window_extent().x0 * 72 / fig.dpi
bad = [(s["text"].strip(), s["bbox"][0]) for b in d[0].get_text("dict")["blocks"]
       for l in b.get("lines", []) for s in l["spans"]
       if s["text"].strip() in leader_y and s["bbox"][0] < x0_axes - 1]
print(f"  labels spilling left of the axes: {len(bad)}" + (f" -> {bad}" if bad else ""))
print(f"  accuracies read from {TABLES.name}: " + ", ".join(f"{k}={v:.2f}" for k, v in acc.items()))
print("saved fig_rtf_quality")
