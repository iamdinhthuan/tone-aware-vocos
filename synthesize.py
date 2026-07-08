from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchaudio

from vocos_train.model import VocosGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy-synthesize an audio file with a trained checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True, help="Output WAV or FLAC path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = VocosGenerator(config["data"], config["model"])
    model.load_state_dict(checkpoint["generator"])
    model.eval().to(device)

    audio, sample_rate = torchaudio.load(args.input)
    audio = audio.mean(dim=0, keepdim=True) if audio.shape[0] > 1 else audio
    target_rate = config["data"]["sample_rate"]
    if sample_rate != target_rate:
        audio = torchaudio.functional.resample(audio, sample_rate, target_rate)
    with torch.inference_mode():
        generated = model(audio.to(device)).float().cpu().clamp(-1, 1)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output), generated, target_rate)
    print(f"saved={output} samples={generated.shape[-1]} sample_rate={target_rate}")


if __name__ == "__main__":
    main()

