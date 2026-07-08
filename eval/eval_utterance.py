"""Utterance-level quality benchmark: UTMOS, PESQ-WB, ESTOI, MCD on full val utterances.

500 utterances sampled (seed 20260706) from mfa_corpus/vi_val10k (24 kHz FLAC, unseen-voice val split).
Systems: 4 internal vocoders + charactr/vocos-mel-24khz + nvidia/bigvgan_v2_24khz_100band_256x.
Per-utterance TSV output for statistics. Run from /data_nvme/vocos_training.
"""
from __future__ import annotations

import argparse
import csv
import glob
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

sys.path.insert(0, "/data_nvme/vocos_training")
SCRATCH = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRATCH / "bigvgan_repo"))

from evaluate_tone_vocos_set import load_generator

SR = 24000


def mcd_db(ref: np.ndarray, syn: np.ndarray, sr: int = SR, order: int = 24) -> float:
    """Mel-cepstral distortion (dB) on time-aligned signals via WORLD + SPTK mcep."""
    import pysptk
    import pyworld

    n = min(len(ref), len(syn))
    ref = ref[:n].astype(np.float64)
    syn = syn[:n].astype(np.float64)
    alpha = 0.466  # all-pass constant for 24 kHz
    frame_period = 5.0
    def mcep(x):
        _f0, t = pyworld.harvest(x, sr, frame_period=frame_period)
        sp = pyworld.cheaptrick(x, _f0, t, sr)
        return pysptk.sp2mc(sp, order=order, alpha=alpha)
    mc_ref = mcep(ref)
    mc_syn = mcep(syn)
    m = min(len(mc_ref), len(mc_syn))
    diff = mc_ref[:m, 1:] - mc_syn[:m, 1:]  # exclude 0th (energy)
    return float((10.0 / np.log(10.0)) * np.sqrt(2.0) * np.mean(np.sqrt(np.sum(diff**2, axis=1))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    from pesq import pesq as pesq_fn
    from pystoi import stoi as stoi_fn
    import librosa

    utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device).eval()

    systems = []
    for name, path in [
        ("baseline", "checkpoints/vocos_mp3/best.pt"),
        ("plus_c", "checkpoints/ablations/plus_c/best.pt"),
        ("plus_cb", "checkpoints/ablations/plus_cb/best.pt"),
        ("plus_cba", "checkpoints/ablations/plus_cba/best.pt"),
    ]:
        gen, _ = load_generator(path, device)
        systems.append((name, lambda x, g=gen: g(x)))

    from vocos import Vocos

    vocos_pre = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device).eval()
    systems.append(("vocos_pretrained_en", lambda x: vocos_pre(x)))

    import bigvgan as bigvgan_mod
    from meldataset import get_mel_spectrogram

    bv = bigvgan_mod.BigVGAN.from_pretrained("nvidia/bigvgan_v2_24khz_100band_256x", use_cuda_kernel=False)
    bv.remove_weight_norm()
    bv = bv.to(device).eval()
    systems.append(("bigvgan_v2", lambda x: bv(get_mel_spectrogram(x.cpu(), bv.h).to(device)).squeeze(1)))

    files = sorted(glob.glob("mfa_corpus/vi_val10k/*.flac"))
    random.Random(args.seed).shuffle(files)
    files = files[: args.n]
    files = [f for i, f in enumerate(files) if i % args.num_shards == args.shard_index]

    out = Path(args.output)
    with out.open("w", encoding="utf-8", newline="") as handle, torch.inference_mode():
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["file", "model", "utmos_ref", "utmos", "pesq_wb", "estoi", "mcd_db", "seconds"])
        for path in tqdm(files, mininterval=30):
            wav, sr = sf.read(path, dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != SR:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
            seconds = len(wav) / SR
            if seconds < 1.0 or seconds > 20.0:
                continue
            x = torch.from_numpy(wav).unsqueeze(0).to(device)
            ref16 = librosa.resample(wav, orig_sr=SR, target_sr=16000)
            utmos_ref = float(utmos(torch.from_numpy(ref16).unsqueeze(0).to(device), 16000).item())
            for name, synth in systems:
                fake = synth(x).float().cpu().squeeze(0).numpy()
                n = min(len(wav), len(fake))
                fake = np.clip(fake[:n], -1.0, 1.0)
                ref = wav[:n]
                fake16 = librosa.resample(fake, orig_sr=SR, target_sr=16000)
                r16 = librosa.resample(ref, orig_sr=SR, target_sr=16000)
                m16 = min(len(r16), len(fake16))
                try:
                    pesq_v = float(pesq_fn(16000, r16[:m16], fake16[:m16], "wb"))
                except Exception:
                    pesq_v = float("nan")
                try:
                    estoi_v = float(stoi_fn(r16[:m16], fake16[:m16], 16000, extended=True))
                except Exception:
                    estoi_v = float("nan")
                try:
                    mcd_v = mcd_db(ref, fake)
                except Exception:
                    mcd_v = float("nan")
                utmos_v = float(utmos(torch.from_numpy(fake16).unsqueeze(0).to(device), 16000).item())
                writer.writerow(
                    [
                        Path(path).name,
                        name,
                        f"{utmos_ref:.4f}",
                        f"{utmos_v:.4f}",
                        f"{pesq_v:.4f}",
                        f"{estoi_v:.4f}",
                        f"{mcd_v:.4f}",
                        f"{seconds:.3f}",
                    ]
                )
            handle.flush()


if __name__ == "__main__":
    main()
