# -*- coding: utf-8 -*-
"""Fig: RTF vs tone accuracy scatter (runbook Phase 6, item 6)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SP = Path(__file__).resolve().parent
FIG = Path("/data_nvme/vocos_training/paper/figures")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e6", "axes.axisbelow": True,
                     "pdf.fonttype": 42})

eff = {r["model"]: r for r in json.load(open(SP / "analysis/efficiency.json"))}
real_acc = json.load(open(SP / "real_acc.json"))["real_tone_accuracy"] * 100
RTF_OURS = eff["ours_vocos_vi"]["rtf_gpu"]

points = [
    ("Vocos-EN", eff["vocos_pretrained_en"]["rtf_gpu"], 57.06, "#E69F00", "s", False),
    ("BigVGAN-v2", eff["bigvgan_v2"]["rtf_gpu"], 58.31, "#009E73", "D", False),
    ("Baseline", RTF_OURS * 0.88, 57.48, "#0072B2", "o", True),
    ("+C", RTF_OURS * 1.13, 57.49, "#56B4E9", "o", True),
    ("+C+B", RTF_OURS, 57.73, "#D55E00", "o", True),
    ("+C+B+A", RTF_OURS, 57.37, "#CC79A7", "o", True),
]

fig, ax = plt.subplots(figsize=(5.4, 2.9))
for name, x, y, c, m, ours in points:
    ax.scatter(x, y, s=46 if ours else 52, color=c, marker=m, zorder=3,
               edgecolor="white", linewidth=0.6)
offsets = {"Vocos-EN": (2, -11), "BigVGAN-v2": (0, 8), "Baseline": (-34, -3),
           "+C": (22, -3), "+C+B": (-28, 4), "+C+B+A": (-36, -6)}
for name, x, y, c, m, ours in points:
    dx, dy = offsets[name]
    ax.annotate(name, (x, y), textcoords="offset points", xytext=(dx, dy),
                fontsize=7.5, ha="center", color="#1a1a1a",
                fontweight="bold" if name == "+C+B" else "normal")
ax.axhline(real_acc, color="black", ls="--", lw=1.0)
ax.text(1.5e-4, real_acc + 0.06, "natural speech (ceiling)", fontsize=7.5)
ax.set_xscale("log")
ax.set_xlabel("Real-time factor on RTX 4090 (log scale, lower = faster)")
ax.set_ylabel("Tone accuracy (%)")
ax.set_xlim(1.2e-4, 3e-2)
ax.set_ylim(56.7, 59.3)
ax.annotate("", xy=(RTF_OURS * 1.55, 57.71), xytext=(RTF_OURS * 1.55, 57.44),
            arrowprops=dict(arrowstyle="->", color="#D55E00", lw=1.2))
ax.text(RTF_OURS * 1.85, 57.58, "same cost,\nbetter tone", fontsize=7.5, color="#D55E00", va="center")
fig.tight_layout()
fig.savefig(FIG / "fig_rtf_quality.pdf", bbox_inches="tight")
fig.savefig(FIG / "fig_rtf_quality.png", dpi=300, bbox_inches="tight")
print("saved fig_rtf_quality")
