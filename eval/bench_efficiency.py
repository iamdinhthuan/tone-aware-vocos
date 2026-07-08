"""Efficiency benchmark: params, GPU RTF, CPU RTF (1 thread), peak VRAM.

Systems: internal Vocos generator (identical across variants), Vocos-EN pretrained, BigVGAN-v2.
Measures analysis-synthesis on real val utterances. Run from /data_nvme/vocos_training.
"""
from __future__ import annotations

import glob
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, "/data_nvme/vocos_training")
SCRATCH = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRATCH / "bigvgan_repo"))

from evaluate_tone_vocos_set import load_generator

SR = 24000


def load_files(n: int) -> list[np.ndarray]:
    files = sorted(glob.glob("mfa_corpus/vi_val10k/*.flac"))
    random.Random(123).shuffle(files)
    out = []
    for f in files:
        wav, sr = sf.read(f, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if 3.0 <= len(wav) / SR <= 12.0:
            out.append(wav)
        if len(out) == n:
            break
    return out


def bench(name, synth, params_m, wavs_gpu, wavs_cpu, device):
    result = {"model": name, "params_M": params_m}
    # GPU
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for w in wavs_gpu[:5]:
                synth(torch.from_numpy(w).unsqueeze(0).to(device), device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            total_audio = 0.0
            per = []
            for w in wavs_gpu:
                x = torch.from_numpy(w).unsqueeze(0).to(device)
                torch.cuda.synchronize()
                s = time.perf_counter()
                synth(x, device)
                torch.cuda.synchronize()
                e = time.perf_counter()
                per.append((e - s) / (len(w) / SR))
                total_audio += len(w) / SR
            t1 = time.perf_counter()
        result["rtf_gpu"] = (t1 - t0) / total_audio
        result["rtf_gpu_mean"] = float(np.mean(per))
        result["rtf_gpu_std"] = float(np.std(per))
        result["vram_MB"] = torch.cuda.max_memory_allocated() / 1e6
    # CPU 1 thread
    torch.set_num_threads(1)
    cpu = torch.device("cpu")
    with torch.inference_mode():
        for w in wavs_cpu[:2]:
            synth(torch.from_numpy(w).unsqueeze(0), cpu)
        per = []
        total_audio = 0.0
        t0 = time.perf_counter()
        for w in wavs_cpu:
            s = time.perf_counter()
            synth(torch.from_numpy(w).unsqueeze(0), cpu)
            e = time.perf_counter()
            per.append((e - s) / (len(w) / SR))
            total_audio += len(w) / SR
        t1 = time.perf_counter()
    result["rtf_cpu"] = (t1 - t0) / total_audio
    result["rtf_cpu_mean"] = float(np.mean(per))
    result["rtf_cpu_std"] = float(np.std(per))
    torch.set_num_threads(torch.get_num_interop_threads() or 8)
    return result


def main() -> None:
    device = torch.device("cuda")
    wavs_gpu = load_files(50)
    wavs_cpu = wavs_gpu[:10]
    results = []

    # internal generator (architecture identical across all 4 variants)
    gen_gpu, _ = load_generator("checkpoints/ablations/plus_cb/best.pt", device)
    gen_cpu, _ = load_generator("checkpoints/ablations/plus_cb/best.pt", torch.device("cpu"))
    p = sum(q.numel() for q in gen_gpu.parameters()) / 1e6

    def synth_ours(x, dev):
        return (gen_gpu if dev.type == "cuda" else gen_cpu)(x)

    results.append(bench("ours_vocos_vi", synth_ours, p, wavs_gpu, wavs_cpu, device))

    from vocos import Vocos

    ven_gpu = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device).eval()
    ven_cpu = Vocos.from_pretrained("charactr/vocos-mel-24khz").eval()
    p = sum(q.numel() for q in ven_gpu.parameters()) / 1e6

    def synth_ven(x, dev):
        return (ven_gpu if dev.type == "cuda" else ven_cpu)(x)

    results.append(bench("vocos_pretrained_en", synth_ven, p, wavs_gpu, wavs_cpu, device))

    import bigvgan as bigvgan_mod
    from meldataset import get_mel_spectrogram

    bv_gpu = bigvgan_mod.BigVGAN.from_pretrained("nvidia/bigvgan_v2_24khz_100band_256x", use_cuda_kernel=False)
    bv_gpu.remove_weight_norm()
    bv_gpu = bv_gpu.to(device).eval()
    bv_cpu = bigvgan_mod.BigVGAN.from_pretrained("nvidia/bigvgan_v2_24khz_100band_256x", use_cuda_kernel=False)
    bv_cpu.remove_weight_norm()
    bv_cpu = bv_cpu.eval()
    p = sum(q.numel() for q in bv_gpu.parameters()) / 1e6

    def synth_bv(x, dev):
        m = bv_gpu if dev.type == "cuda" else bv_cpu
        mel = get_mel_spectrogram(x.cpu(), m.h).to(dev)
        return m(mel)

    results.append(bench("bigvgan_v2", synth_bv, p, wavs_gpu, wavs_cpu, device))

    out = SCRATCH / "analysis" / "efficiency.json"
    out.write_text(json.dumps(results, indent=2))
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
