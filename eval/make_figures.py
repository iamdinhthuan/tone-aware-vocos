"""Publication figures for the EAAI paper. Writes PDF+PNG into /data_nvme/vocos_training/paper/figures.

Palette validated with dataviz validator (Okabe-Ito subset):
  Vocos-EN #E69F00 (hatch), BigVGAN-v2 #009E73 (hatch),
  baseline #0072B2, +C #56B4E9, +C+B #D55E00 (hero), +C+B+A #CC79A7.
Run from /data_nvme/vocos_training with py310.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SP = Path(__file__).resolve().parent
FIG = Path("/data_nvme/vocos_training/paper/figures")
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e6e6",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "pdf.fonttype": 42,
    }
)

COLORS = {
    "vocos_pretrained_en": "#E69F00",
    "bigvgan_v2": "#009E73",
    "baseline": "#0072B2",
    "plus_c": "#56B4E9",
    "plus_cb": "#D55E00",
    "plus_cba": "#CC79A7",
}
HATCH = {"vocos_pretrained_en": "//", "bigvgan_v2": "//"}
LABEL = {
    "vocos_pretrained_en": "Vocos-EN",
    "bigvgan_v2": "BigVGAN-v2",
    "baseline": "Baseline (ours)",
    "plus_c": "+C (ours)",
    "plus_cb": "+C+B (ours)",
    "plus_cba": "+C+B+A (ours)",
}
ORDER = ["vocos_pretrained_en", "bigvgan_v2", "baseline", "plus_c", "plus_cb", "plus_cba"]
TONES = ["ngang", "sac", "huyen", "hoi", "nga", "nang"]
TONE_LABEL = ["ngang", "sắc", "huyền", "hỏi", "ngã", "nặng"]

REAL = json.load(open(SP / "real_acc.json"))


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def fig_per_tone():
    acc = pd.read_csv(SP / "per_tone_acc.tsv", sep="\t").set_index("model")
    f0 = pd.read_csv(SP / "per_tone_f0.tsv", sep="\t").set_index("model")
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.6))
    x = np.arange(len(TONES))
    n = len(ORDER)
    width = 0.8 / n
    for ax, table, ylab in [
        (axes[0], f0, "F0 RMSE (Hz)"),
        (axes[1], acc * 100, "Tone accuracy (%)"),
    ]:
        for i, m in enumerate(ORDER):
            vals = [table.loc[m, t] for t in TONES]
            ax.bar(
                x + (i - n / 2 + 0.5) * width,
                vals,
                width * 0.92,
                color=COLORS[m],
                hatch=HATCH.get(m, ""),
                edgecolor="white",
                linewidth=0.4,
                label=LABEL[m],
            )
        ax.set_xticks(x)
        ax.set_xticklabels(TONE_LABEL)
        ax.set_ylabel(ylab)
    # real-speech ceiling on accuracy panel
    for j, t in enumerate(TONES):
        lo = x[j] - 0.44
        hi = x[j] + 0.44
        axes[1].plot([lo, hi], [REAL["per_tone"][j] * 100] * 2, color="black", ls="--", lw=1.0)
    axes[1].plot([], [], color="black", ls="--", lw=1.0, label="Source recordings (ceiling)")
    axes[1].legend(ncol=4, loc="upper right", frameon=False, bbox_to_anchor=(1.0, 1.28))
    axes[0].set_title("Per-tone analysis on 20,000 unseen-voice syllables", loc="left", pad=4)
    fig.tight_layout()
    save(fig, "fig_per_tone")


def fig_confusion():
    conf_real = np.array(REAL["confusion"], dtype=float)
    conf = {"Source recordings": conf_real}
    for name, mdl in [("Baseline", "baseline"), ("+C+B", "plus_cb")]:
        d = json.load(open(f"eval_reports/tone_val_no_overlap_{mdl}_20k.json"))
        conf[name] = np.array(d["confusion"], dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.55))
    for ax, (title, C) in zip(axes, conf.items()):
        R = C / C.sum(axis=1, keepdims=True)
        im = ax.imshow(R, cmap="Blues", vmin=0, vmax=0.9)
        ax.set_xticks(range(6))
        ax.set_yticks(range(6))
        ax.set_xticklabels(TONE_LABEL, rotation=45, ha="right")
        ax.set_yticklabels(TONE_LABEL if ax is axes[0] else [])
        ax.set_title(title)
        ax.grid(False)
        for i in range(6):
            for j in range(6):
                if R[i, j] >= 0.06:
                    ax.text(
                        j,
                        i,
                        f"{R[i,j]:.2f}".lstrip("0"),
                        ha="center",
                        va="center",
                        fontsize=6.5,
                        color="white" if R[i, j] > 0.5 else "#1a1a1a",
                    )
    axes[0].set_ylabel("True tone")
    axes[1].set_xlabel("Predicted tone")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="Row-normalized rate")
    save(fig, "fig_confusion")


def fig_summary():
    s = pd.read_csv(SP / "segment_summary.tsv", sep="\t").set_index("model")
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    metrics = [
        ("f0_rmse", "F0 RMSE (Hz) ↓", 1.0),
        ("vuv_error", "V/UV error (%) ↓", 100.0),
        ("correct", "Tone accuracy (%) ↑", 100.0),
    ]
    x = np.arange(len(ORDER))
    for ax, (m, title, scale) in zip(axes, metrics):
        vals = np.array([s.loc[mm, m] for mm in ORDER]) * scale
        los = np.array([s.loc[mm, f"{m}_lo"] for mm in ORDER]) * scale
        his = np.array([s.loc[mm, f"{m}_hi"] for mm in ORDER]) * scale
        for i, mm in enumerate(ORDER):
            ax.bar(
                i,
                vals[i],
                0.72,
                color=COLORS[mm],
                hatch=HATCH.get(mm, ""),
                edgecolor="white",
                linewidth=0.4,
            )
        ax.errorbar(x, vals, yerr=[vals - los, his - vals], fmt="none", ecolor="#333333", capsize=2, lw=0.9)
        if m == "correct":
            ax.axhline(REAL["real_tone_accuracy"] * 100, color="black", ls="--", lw=1.0)
            ax.text(0.03, REAL["real_tone_accuracy"] * 100 + 0.15, "source recordings", fontsize=7)
            ax.set_ylim(55, 60)
        ax.set_xticks(x)
        ax.set_xticklabels([LABEL[mm].replace(" (ours)", "") for mm in ORDER], rotation=38, ha="right")
        ax.set_title(title, loc="left")
    fig.tight_layout()
    save(fig, "fig_summary")


def fig_contours_and_specs():
    """F0 contour overlays and spectrogram close-ups for glottalized-tone segments."""
    import torch
    import librosa

    sys.path.insert(0, "/data_nvme/vocos_training")
    from evaluate_tone_vocos import load_segment
    from evaluate_tone_vocos_set import load_generator, extract_f0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gens = {}
    for name, path in [("baseline", "checkpoints/vocos_mp3/best.pt"), ("plus_cb", "checkpoints/ablations/plus_cb/best.pt")]:
        gens[name], _ = load_generator(path, device)

    # pick nga/nang segments where +CB clearly improves over baseline
    frames = [pd.read_csv(f, sep="\t") for f in sorted(SP.glob("peritem_shard*.tsv"))]
    df = pd.concat(frames, ignore_index=True)
    df["dur"] = df["end"] - df["start"]
    piv = df.pivot_table(index=["audio", "start", "end", "tone", "dur"], columns="model", values="f0_rmse").reset_index()
    cand = piv[(piv["tone"].isin([4, 5])) & (piv["dur"] >= 0.30) & piv["baseline"].notna() & piv["plus_cb"].notna()]
    cand = cand.assign(gain=cand["baseline"] - cand["plus_cb"]).sort_values("gain", ascending=False)
    picks = cand.head(3)

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    hop_t = 256 / 24000
    for ax, (_, row) in zip(axes, picks.iterrows()):
        real = load_segment(row["audio"], float(row["start"]), float(row["end"]), 24000)
        x = real.unsqueeze(0).to(device)
        t = None
        for name, color, label in [
            (None, "black", "Source"),
            ("baseline", COLORS["baseline"], "Baseline"),
            ("plus_cb", COLORS["plus_cb"], "+C+B"),
        ]:
            if name is None:
                wav = real.numpy()
            else:
                with torch.inference_mode():
                    wav = gens[name](x).float().cpu().squeeze(0).numpy()
            f0, voiced = extract_f0(wav, 24000)
            f0 = np.where(voiced, f0, np.nan)
            t = np.arange(len(f0)) * hop_t * 1000
            ax.plot(t, f0, color=color, lw=1.4 if name is None else 1.1, label=label, ls="-" if name is None else "-")
        tone_name = TONE_LABEL[int(row["tone"])]
        ax.set_title(f"tone {tone_name}", loc="left")
        ax.set_xlabel("Time (ms)")
    axes[0].set_ylabel("F0 (Hz)")
    axes[0].legend(frameon=False, loc="best")
    fig.tight_layout()
    save(fig, "fig_f0_contours")

    # spectrogram close-up of the top pick
    row = picks.iloc[0]
    real = load_segment(row["audio"], float(row["start"]), float(row["end"]), 24000)
    x = real.unsqueeze(0).to(device)
    sigs = [("Source", real.numpy())]
    for name, lbl in [("baseline", "Baseline"), ("plus_cb", "+C+B")]:
        with torch.inference_mode():
            sigs.append((lbl, gens[name](x).float().cpu().squeeze(0).numpy()))
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), sharey=True)
    for ax, (lbl, wav) in zip(axes, sigs):
        S = librosa.amplitude_to_db(np.abs(librosa.stft(wav, n_fft=1024, hop_length=128)), ref=np.max)
        freqs = librosa.fft_frequencies(sr=24000, n_fft=1024)
        keep = freqs <= 2000
        extent = [0, len(wav) / 24000 * 1000, 0, 2000]
        ax.imshow(S[keep], aspect="auto", origin="lower", extent=extent, cmap="magma", vmin=-70, vmax=0)
        ax.set_title(lbl, loc="left")
        ax.set_xlabel("Time (ms)")
        ax.grid(False)
    axes[0].set_ylabel("Frequency (Hz)")
    tone_name = TONE_LABEL[int(row["tone"])]
    fig.suptitle(f"Glottalized syllable (tone {tone_name}), 0–2 kHz", y=1.03, fontsize=9.5)
    save(fig, "fig_spectrogram")


if __name__ == "__main__":
    fig_per_tone()
    fig_confusion()
    fig_summary()
    fig_contours_and_specs()
