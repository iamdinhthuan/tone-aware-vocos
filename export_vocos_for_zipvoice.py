from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export this repo's Vocos checkpoint to ZipVoice/upstream Vocos format")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    data, model = config["data"], config["model"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hparams = {
        "feature_extractor": {
            "class_path": "vocos.feature_extractors.MelSpectrogramFeatures",
            "init_args": {
                "sample_rate": data["sample_rate"],
                "n_fft": model["n_fft"],
                "hop_length": model["hop_length"],
                "n_mels": model["n_mels"],
                "padding": "center",
            },
        },
        "backbone": {
            "class_path": "vocos.models.VocosBackbone",
            "init_args": {
                "input_channels": model["n_mels"],
                "dim": model["dim"],
                "intermediate_dim": model["intermediate_dim"],
                "num_layers": model["num_layers"],
            },
        },
        "head": {
            "class_path": "vocos.heads.ISTFTHead",
            "init_args": {
                "dim": model["dim"],
                "n_fft": model["n_fft"],
                "hop_length": model["hop_length"],
                "padding": "center",
            },
        },
    }
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(hparams, handle, sort_keys=False)
    torch.save(checkpoint["generator"], output_dir / "pytorch_model.bin")
    print(f"exported={output_dir} step={checkpoint.get('step')} val_loss={checkpoint.get('val_loss', checkpoint.get('best_val'))}")


if __name__ == "__main__":
    main()
