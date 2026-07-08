from __future__ import annotations

import argparse

from vocos_train.config import load_config
from vocos_train.trainer import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Vocos directly from MP3 files")
    parser.add_argument("--config", default="configs/vocos_mp3.yaml")
    parser.add_argument("--resume", help="Checkpoint path, usually checkpoints/vocos_mp3/last.pt")
    parser.add_argument("--device", help="Torch device such as cuda, cuda:0, or cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(load_config(args.config), resume=args.resume, device_name=args.device)

