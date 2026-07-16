from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")

    for section in ("data", "model", "training"):
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"Missing config section: {section}")
    config.setdefault("tone_aware", {})

    base = next((parent for parent in (path.parent, *path.parents) if (parent / "pyproject.toml").is_file()), path.parent)
    for key in ("train_manifest", "val_manifest"):
        value = Path(config["data"][key]).expanduser()
        if not value.is_absolute():
            value = base / value
        config["data"][key] = str(value.resolve())
    for key in ("output_dir", "log_dir"):
        value = Path(config["training"][key]).expanduser()
        if not value.is_absolute():
            value = base / value
        config["training"][key] = str(value.resolve())
    if "tone_classifier_ckpt" in config["tone_aware"]:
        value = Path(config["tone_aware"]["tone_classifier_ckpt"]).expanduser()
        if not value.is_absolute():
            value = base / value
        config["tone_aware"]["tone_classifier_ckpt"] = str(value.resolve())
    config["_config_path"] = str(path)
    return config


def validate_config(config: dict[str, Any]) -> None:
    data = config["data"]
    model = config["model"]
    training = config["training"]
    tone_aware = config.get("tone_aware", {})

    if model["dim"] <= 0 or model["num_layers"] <= 0:
        raise ValueError("Model dimensions must be positive")
    if data["sample_rate"] <= 0 or data["train_num_samples"] <= 0:
        raise ValueError("Sample rate and segment length must be positive")
    if data["train_num_samples"] % model["hop_length"] != 0:
        raise ValueError("train_num_samples must be divisible by hop_length")
    if data["val_num_samples"] % model["hop_length"] != 0:
        raise ValueError("val_num_samples must be divisible by hop_length")
    if training["precision"] not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision must be one of: fp32, fp16, bf16")
    if training["pretrain_mel_steps"] >= training["max_steps"]:
        raise ValueError("pretrain_mel_steps must be smaller than max_steps")
    if int(training.get("save_top_k_best", 3)) <= 0:
        raise ValueError("save_top_k_best must be positive")
    if int(training.get("keep_last_checkpoints", 3)) <= 0:
        raise ValueError("keep_last_checkpoints must be positive")
    if tone_aware.get("use_tone_loss", False):
        checkpoint = Path(tone_aware["tone_classifier_ckpt"])
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Tone classifier checkpoint not found: {checkpoint}")
    if float(tone_aware.get("stft_loss_coeff", 1.0)) < 0:
        raise ValueError("stft_loss_coeff must be non-negative")
    if float(tone_aware.get("tone_loss_coeff", 1.0)) < 0:
        raise ValueError("tone_loss_coeff must be non-negative")
