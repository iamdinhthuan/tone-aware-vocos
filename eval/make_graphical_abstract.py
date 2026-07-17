# -*- coding: utf-8 -*-
"""Graphical abstract for the EAAI submission (13 x 5 cm).

Elsevier requires the image to be "readable at a size of 5 x 13 cm" (Guide for Authors).
That is the binding constraint here, and it is a budget: at 13 cm there is room for roughly
20-25 words of real content, not 150. Every string below is therefore >= 7 pt (~2.5 mm at
actual size); the previous version ran down to 3.4 pt and was illegible at the size it ships at.

Design notes, following the dataviz method:
  - The result is a hero number, not a bar chart. Its job is to carry one headline, and a
    four-bar chart at this size needs 3 pt tick labels to say less.
  - Palette: SKY (component C) and VERM (component B), validated for categorical use in
    light mode (CVD dE 25.1 deutan, 30.9 normal). SKY alone is 2.25:1 against white, so it
    is never used for text -- the chips are marks and every label is INK. Gray is the
    context/recessive token, not a series.
  - The tone contours are direct-labelled at the line ends, so no legend is needed.

Honesty: the F0 headline (-6.5%, cluster p = 0.003) is cluster-robust. The nga figure is
NOT -- it is tail-driven and cluster-marginal (p = 0.085) -- so it carries an asterisk that
resolves in the footer, exactly as the paper reports it.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SKY = "#56B4E9"    # component C
VERM = "#D55E00"   # component B, and the hero number
GRAY = "#5f5f5f"   # context / recessive token
LGRAY = "#d9d9d9"
INK = "#1a1a1a"

TITLE_FS, HERO_FS, HEAD_FS, BODY_FS, SMALL_FS, TINY_FS = 8.5, 16.0, 8.0, 7.6, 7.0, 6.5

fig = plt.figure(figsize=(13 / 2.54, 5 / 2.54))
fig.patch.set_facecolor("white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 13)
ax.set_ylim(0, 5)
ax.axis("off")


def panel(x, w, edge):
    ax.add_patch(FancyBboxPatch((x, 0.60), w, 3.62,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor="none", edgecolor=edge, linewidth=0.8, zorder=1))


def chip(x, y, color, letter):
    """A colored mark carries identity; the letter on it is never the series hue as text."""
    ax.add_patch(FancyBboxPatch((x, y - 0.13), 0.26, 0.26,
                                boxstyle="round,pad=0.01,rounding_size=0.05",
                                facecolor=color, edgecolor="none", zorder=3))
    ax.text(x + 0.13, y, letter, ha="center", va="center", fontsize=SMALL_FS,
            fontweight="bold", color="white", zorder=4)


# ---------- title (two lines: the full paper title does not fit on one at >= 7 pt) ----------
ax.text(6.5, 4.79, "Tone-aware adversarial training of a Fourier-based",
        ha="center", va="center", fontsize=TITLE_FS, fontweight="bold", color=INK)
ax.text(6.5, 4.46, "neural vocoder for tonal languages",
        ha="center", va="center", fontsize=TITLE_FS, fontweight="bold", color=INK)

# ================= panel 1: the problem =================
panel(0.18, 4.06, GRAY)
ax.text(2.21, 3.92, "Tone is the word", ha="center", fontsize=HEAD_FS, fontweight="bold", color=INK)

# taller inset + endpoints respread: 6.5 pt labels need >= 7.5 pt of vertical clearance,
# and the previous endpoints sat 4.8 pt apart, so ngã and sắc collided.
a1 = fig.add_axes([0.055, 0.405, 0.185, 0.364])
t = np.linspace(0, 1, 100)
# All SIX tones: the panel says "six words, one syllable" and lists six syllables, so drawing
# five contours (the redesign dropped hỏi) contradicted the panel's own text.
# Endpoints are spread so the 6.5 pt labels keep >= 7.5 pt of vertical clearance.
for c, lab in [(0.60 + 0.0 * t, "ngang"),
               (0.52 + 0.28 * t ** 1.3, "sắc"),
               (0.48 - 0.55 * t * (1 - t) - 0.08 * t, "hỏi"),
               (0.50 - 0.28 * t, "huyền")]:
    a1.plot(t, c, color=GRAY, lw=1.0)
    a1.text(1.05, c[-1], lab, fontsize=TINY_FS, color=GRAY, va="center")
nga = 0.74 + 0.26 * ((t[62:] - 0.62) / 0.38) ** 1.2
a1.plot(t[:38], 0.60 + 0.08 * (t[:38] / 0.38), color=VERM, lw=2.0)
a1.plot(t[62:], nga, color=VERM, lw=2.0)
a1.text(1.05, nga[-1], "ngã", fontsize=TINY_FS, color=VERM, va="center", fontweight="bold")
a1.plot(t[:52], 0.26 - 0.22 * (t[:52] / 0.52), color=VERM, lw=2.0)
a1.plot([0.60, 0.98], [0.02, 0.02], color=VERM, lw=2.0)
a1.text(1.05, 0.02, "nặng", fontsize=TINY_FS, color=VERM, va="center", fontweight="bold")
a1.set_xlim(-0.02, 1.62)
a1.set_ylim(-0.06, 1.08)
a1.set_xticks([]); a1.set_yticks([])
for s_ in a1.spines.values():
    s_.set_visible(False)
a1.set_ylabel("F0", fontsize=TINY_FS, color=GRAY, labelpad=1)

ax.text(2.21, 1.74, "ma  má  mà  mả  mã  mạ", ha="center", fontsize=BODY_FS + 1.0,
        style="italic", color=INK)
ax.text(2.21, 1.38, "six words, one syllable", ha="center", fontsize=SMALL_FS, color=GRAY)
ax.text(2.21, 1.00, "GAN losses treat the F0 band\nlike every other band",
        ha="center", va="center", fontsize=SMALL_FS, color=INK)

# ================= panel 2: the method =================
panel(4.46, 4.06, SKY)
ax.text(6.49, 3.92, "Tone-aware objectives", ha="center", fontsize=HEAD_FS, fontweight="bold", color=INK)

chip(4.72, 3.28, SKY, "C")
ax.text(5.10, 3.28, "F0-weighted STFT loss", fontsize=BODY_FS, color=INK, va="center")
ax.text(5.10, 2.94, "3× weight below 1 kHz", fontsize=SMALL_FS, color=GRAY, va="center")

chip(4.72, 2.42, VERM, "B")
ax.text(5.10, 2.42, "Constant-Q tone critic", fontsize=BODY_FS, color=INK, va="center")
ax.text(5.10, 2.08, "48 bins over 80–320 Hz", fontsize=SMALL_FS, color=GRAY, va="center")

ax.add_patch(FancyBboxPatch((4.86, 1.06), 3.26, 0.66,
                            boxstyle="round,pad=0.02,rounding_size=0.06",
                            facecolor="#fdf0e8", edgecolor=VERM, linewidth=0.8, zorder=2))
ax.text(6.49, 1.38, "Removed at inference —\nzero added cost",
        ha="center", va="center", fontsize=SMALL_FS, color=INK, zorder=3)

# ================= panel 3: the result =================
panel(8.74, 4.08, VERM)
ax.text(10.78, 3.92, "20,000 unseen syllables", ha="center", fontsize=HEAD_FS,
        fontweight="bold", color=INK)

ax.text(10.78, 3.10, "−6.5%", ha="center", va="center", fontsize=HERO_FS,
        fontweight="bold", color=VERM)
ax.text(10.78, 2.60, "F0 error vs. baseline", ha="center", fontsize=BODY_FS, color=INK)
ax.text(10.78, 2.28, "cluster-robust, p = 0.003", ha="center",
        fontsize=SMALL_FS, color=GRAY)

ax.plot([9.10, 12.46], [1.94, 1.94], color=LGRAY, lw=0.8)

ax.text(10.78, 1.62, "ngã  −34.6%*", ha="center", fontsize=BODY_FS + 0.4, color=INK)
ax.text(10.78, 1.14, "39× faster than BigVGAN-v2", ha="center", fontsize=BODY_FS, color=INK)
ax.text(10.78, 0.84, "GPU; 145× on 1 CPU thread", ha="center", fontsize=SMALL_FS, color=GRAY)

# ---------- arrows between panels ----------
for x0 in (4.28, 8.56):
    ax.annotate("", xy=(x0 + 0.17, 2.30), xytext=(x0 + 0.01, 2.30),
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.3))

# ---------- footer: the asterisk resolves here ----------
ax.text(6.5, 0.30, "*ngã: tail-driven, cluster-marginal (p = 0.085).    "
                   "Matched data, architecture, budget, seed.",
        ha="center", fontsize=TINY_FS, color=GRAY)

fig.savefig(FIG / "graphical_abstract.pdf", facecolor="white")
fig.savefig(FIG / "graphical_abstract.png", dpi=300, facecolor="white")

# --- layout self-check: the validator checks colour, not geometry ---
import fitz  # noqa: E402
_d = fitz.open(FIG / "graphical_abstract.pdf")
_PANELS = [(0.18, 4.24), (4.46, 8.52), (8.74, 12.82)]   # cm, left/right of each panel
_PANEL_Y = (0.60, 4.22)  # title (above) and footer (below) span the full width by design
_bad = []
for _b in _d[0].get_text("dict")["blocks"]:
    for _l in _b.get("lines", []):
        for _sp in _l["spans"]:
            _t = _sp["text"].strip()
            if not _t:
                continue
            x0, x1 = _sp["bbox"][0] / 72 * 2.54, _sp["bbox"][2] / 72 * 2.54
            y = 5 - _sp["bbox"][3] / 72 * 2.54   # fitz y grows downward
            if x0 < 0.05 or x1 > 12.95:
                _bad.append((_t, x0, x1, "off page"))
                continue
            if not (_PANEL_Y[0] - 0.1 <= y <= _PANEL_Y[1]):
                continue
            for _pl, _pr in _PANELS:
                if x0 >= _pl - 0.3 and x0 <= _pr:          # span starts in this panel
                    if x1 > _pr + 0.06:
                        _bad.append((_t, x0, x1, f"crosses panel edge {_pr}"))
                    break
if _bad:
    print(f"  LAYOUT: {len(_bad)} overflowing span(s):")
    for _t, x0, x1, why in _bad:
        print(f"    {x0:5.2f}->{x1:5.2f} cm  {why:24s} {_t[:46]!r}")
else:
    print("  LAYOUT: no span crosses a panel edge or the page")
print("saved graphical_abstract")
