# -*- coding: utf-8 -*-
"""Graphical abstract for the EAAI submission (13 x 5 cm @ 300 dpi).

Single master axes in cm coordinates (0-13 x 0-5); inset axes for mini plots.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

FIG = Path("/data_nvme/vocos_training/paper/figures")

BLUE = "#0072B2"
SKY = "#56B4E9"
VERM = "#D55E00"
GREEN = "#009E73"
ORANGE = "#E69F00"
PURP = "#CC79A7"
GRAY = "#5f5f5f"
LGRAY = "#e9e9e9"
INK = "#1a1a1a"

TITLE_FS, HEAD_FS, BODY_FS, SMALL_FS = 7.0, 5.6, 4.6, 4.0

fig = plt.figure(figsize=(13 / 2.54, 5 / 2.54), dpi=300)
fig.patch.set_facecolor("white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 13)
ax.set_ylim(0, 5)
ax.axis("off")


def rbox(x, y, w, h, edge, face="none", lw=0.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.07",
                                facecolor=face, edgecolor=edge, linewidth=lw, zorder=1))


def inset(x, y, w, h):
    return fig.add_axes([x / 13, y / 5, w / 13, h / 5])


# ---------- title ----------
ax.text(6.5, 4.78, "Tone-aware training of a Fourier-based neural vocoder for Vietnamese TTS",
        ha="center", va="center", fontsize=TITLE_FS, fontweight="bold", color=INK)

PT, PB = 4.35, 0.18   # panel top / bottom
# ---------- panel 1: problem ----------
rbox(0.12, PB, 3.40, PT - PB, GRAY)
ax.text(1.82, 4.08, "Problem: tone = meaning", ha="center", fontsize=HEAD_FS, fontweight="bold", color=INK)

a1 = inset(0.42, 1.72, 2.45, 1.95)
t = np.linspace(0, 1, 100)
for name, c, col, lw in [
    ("ngang", 0.62 + 0.0 * t, GRAY, 0.9),
    ("sắc", 0.52 + 0.40 * t ** 1.3, GRAY, 0.9),
    ("huyền", 0.55 - 0.28 * t, GRAY, 0.9),
    ("hỏi", 0.50 - 0.55 * t * (1 - t), GRAY, 0.9),
]:
    a1.plot(t, c, color=col, lw=lw)
    a1.text(1.03, c[-1], name, fontsize=SMALL_FS, color=INK, va="center")
nga2 = 0.72 + 0.45 * ((t[62:] - 0.62) / 0.38) ** 1.2
a1.plot(t[:38], 0.58 + 0.10 * (t[:38] / 0.38), color=VERM, lw=1.3)
a1.plot(t[62:], nga2, color=VERM, lw=1.3)
a1.text(1.03, nga2[-1], "ngã", fontsize=SMALL_FS, color=VERM, va="center", fontweight="bold")
a1.plot(t[:52], 0.38 - 0.24 * (t[:52] / 0.52), color=VERM, lw=1.3)
a1.plot([0.56, 0.64], [0.13, 0.13], color=VERM, lw=1.3)
a1.text(1.03, 0.13, "nặng", fontsize=SMALL_FS, color=VERM, va="center", fontweight="bold")
a1.text(0.42, 1.10, "glottalized (creaky)", fontsize=SMALL_FS, color=VERM, ha="center")
a1.set_xlim(-0.02, 1.42)
a1.set_ylim(0.02, 1.22)
a1.set_xticks([]); a1.set_yticks([])
for s in a1.spines.values():
    s.set_visible(False)
a1.set_ylabel("F0", fontsize=SMALL_FS, labelpad=0)

ax.text(1.82, 1.34, "ma  má  mà  mả  mã  mạ", ha="center", fontsize=BODY_FS + 0.4, style="italic", color=INK)
ax.text(1.82, 1.06, "one syllable, six words", ha="center", fontsize=SMALL_FS, color=GRAY)
ax.text(1.82, 0.62, "GAN vocoder losses treat the\nnarrow F0 band like any other band",
        ha="center", va="center", fontsize=SMALL_FS, color=INK)

# ---------- panel 2: method ----------
rbox(3.72, PB, 4.85, PT - PB, BLUE)
ax.text(6.14, 4.08, "Tone-aware objectives — training-time only", ha="center", fontsize=HEAD_FS,
        fontweight="bold", color=INK)

gy = 3.30
chain = [("audio", 3.92, 0.62, LGRAY), ("mel-100", 4.68, 0.72, LGRAY),
         ("ConvNeXt ×8", 5.54, 1.10, LGRAY), ("iSTFT", 6.78, 0.62, LGRAY), ("ŷ", 7.54, 0.40, "white")]
for label, bx, bw, fc in chain:
    rbox(bx, gy - 0.22, bw, 0.44, GRAY, face=fc, lw=0.7)
    ax.text(bx + bw / 2, gy, label, ha="center", va="center", fontsize=SMALL_FS, color=INK)
for x0, x1 in [(4.54, 4.68), (5.40, 5.54), (6.64, 6.78), (7.40, 7.54)]:
    ax.annotate("", xy=(x1, gy), xytext=(x0, gy), arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.7))
ax.text(6.14, 3.72, "Vocos generator — unchanged at inference (13.5 M params)",
        ha="center", fontsize=SMALL_FS, color=GRAY, style="italic")

badges = [
    ("C", "F0-weighted MR-STFT loss — 3× weight below 1 kHz", SKY),
    ("B", "Constant-Q tone critic — 48 log-freq bins, 80–320 Hz", VERM),
    ("A", "frozen tone-classifier feature loss — negative ablation", PURP),
]
by = 2.52
for tag, label, color in badges:
    rbox(3.95, by - 0.26, 4.40, 0.52, color, lw=0.9)
    ax.text(4.14, by, tag, fontsize=BODY_FS + 0.6, color=color, fontweight="bold", va="center", ha="center")
    ax.text(4.38, by, label, fontsize=SMALL_FS + 0.2, color=INK, va="center")
    by -= 0.66
ax.annotate("", xy=(6.14, 2.84), xytext=(6.14, 3.04),
            arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.8))
ax.text(6.14, 0.50, "removed at deployment  →  zero added inference cost",
        ha="center", fontsize=SMALL_FS + 0.4, color=VERM, fontweight="bold")

# ---------- panel 3: results ----------
rbox(8.92, PB, 3.96, PT - PB, VERM)
ax.text(10.90, 4.08, "20,000 unseen-voice syllables", ha="center", fontsize=HEAD_FS, fontweight="bold", color=INK)

a3 = inset(9.30, 1.55, 1.55, 2.0)
vals = [3.633, 2.282, 3.498, 3.271]
cols = [ORANGE, GREEN, BLUE, VERM]
names = ["Vocos-EN", "BigVGAN-v2", "Baseline", "+C+B"]
bars = a3.bar(range(4), vals, 0.62, color=cols, edgecolor="white", lw=0.3)
bars[0].set_hatch("///")
bars[1].set_hatch("///")
a3.set_xticks(range(4))
a3.set_xticklabels(names, fontsize=3.4, rotation=38, ha="right")
a3.set_yticks([])
for s in a3.spines.values():
    s.set_visible(False)
a3.set_ylim(0, 5.1)
a3.set_title("F0 RMSE (Hz) ↓", fontsize=SMALL_FS + 0.2, pad=1)
a3.annotate("−6.5%*\nngã −34.6%", xy=(3.05, 3.42), xytext=(1.85, 4.45), fontsize=SMALL_FS,
            color=VERM, fontweight="bold", ha="center", va="center",
            arrowprops=dict(arrowstyle="->", color=VERM, lw=0.6))

fx = 11.02
facts = [
    ("98%", "of natural-speech\ntone accuracy"),
    ("≈ BigVGAN-v2", "tone accuracy at\n12% of the params"),
    ("39× / 145×", "faster than BigVGAN-v2\n(GPU / 1 CPU thread)"),
    ("PESQ ↑", "utterance quality\npreserved"),
]
fy = 3.50
for head, sub in facts:
    ax.text(fx, fy, head, fontsize=BODY_FS + 0.8, color=VERM, fontweight="bold", va="center")
    ax.text(fx, fy - 0.34, sub, fontsize=SMALL_FS, color=INK, va="center")
    fy -= 0.80
ax.text(10.90, 0.42, "*paired Wilcoxon + Holm–Bonferroni", ha="center", fontsize=SMALL_FS, color=GRAY)

# ---------- arrows between panels ----------
for x0 in (3.54, 8.60):
    ax.annotate("", xy=(x0 + 0.32, 2.25), xytext=(x0, 2.25),
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.4))

fig.savefig(FIG / "graphical_abstract.pdf", facecolor="white")
fig.savefig(FIG / "graphical_abstract.png", dpi=300, facecolor="white")
print("saved")
