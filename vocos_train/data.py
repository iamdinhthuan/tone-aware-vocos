from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset


class AudioDataset(Dataset[torch.Tensor]):
    def __init__(
        self,
        manifest_path: str,
        audio_root: str,
        sample_rate: int,
        num_samples: int,
        train: bool,
        max_decode_retries: int = 8,
    ) -> None:
        manifest = Path(manifest_path)
        if not manifest.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest}")
        with manifest.open("r", encoding="utf-8") as handle:
            self.files = [line.strip() for line in handle if line.strip()]
        if not self.files:
            raise ValueError(f"Manifest is empty: {manifest}")
        self.audio_root = Path(audio_root).expanduser()
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.train = train
        self.max_decode_retries = max_decode_retries

    def __len__(self) -> int:
        return len(self.files)

    def _path(self, index: int) -> Path:
        path = Path(self.files[index])
        return path if path.is_absolute() else self.audio_root / path

    def _load(self, index: int) -> torch.Tensor:
        path = self._path(index)
        waveform, sample_rate = torchaudio.load(str(path))
        if waveform.numel() == 0:
            raise RuntimeError("decoded waveform is empty")
        waveform = waveform.float()
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        waveform = waveform[0]
        if not torch.isfinite(waveform).all():
            raise RuntimeError("decoded waveform contains NaN or Inf")

        peak = waveform.abs().amax().clamp_min(1e-7)
        gain_db = torch.empty(()).uniform_(-6.0, -1.0).item() if self.train else -3.0
        waveform = waveform * ((10.0 ** (gain_db / 20.0)) / peak)

        length = waveform.numel()
        if length < self.num_samples:
            if self.train:
                repeats = (self.num_samples + length - 1) // length
                waveform = waveform.repeat(repeats)[: self.num_samples]
            else:
                waveform = torch.nn.functional.pad(waveform, (0, self.num_samples - length))
        elif length > self.num_samples:
            if self.train:
                start = torch.randint(0, length - self.num_samples + 1, ()).item()
            else:
                start = (length - self.num_samples) // 2
            waveform = waveform[start : start + self.num_samples]
        return waveform.contiguous()

    def __getitem__(self, index: int) -> torch.Tensor:
        errors: list[str] = []
        for attempt in range(self.max_decode_retries):
            candidate = (index + attempt * 104729) % len(self.files)
            try:
                return self._load(candidate)
            except Exception as exc:  # A rare damaged MP3 must not kill a long run.
                errors.append(f"{self._path(candidate)}: {exc}")
        raise RuntimeError("Could not decode audio after retries:\n" + "\n".join(errors))


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def create_dataloaders(config: dict, device: torch.device) -> tuple[DataLoader, DataLoader]:
    data = config["data"]
    common = {
        "audio_root": data["audio_root"],
        "sample_rate": data["sample_rate"],
        "max_decode_retries": data.get("max_decode_retries", 8),
    }
    train_dataset = AudioDataset(
        manifest_path=data["train_manifest"],
        num_samples=data["train_num_samples"],
        train=True,
        **common,
    )
    val_dataset = AudioDataset(
        manifest_path=data["val_manifest"],
        num_samples=data["val_num_samples"],
        train=False,
        **common,
    )
    workers = int(data["num_workers"])
    generator = torch.Generator().manual_seed(int(config["seed"]))
    loader_common = {
        "batch_size": int(data["batch_size"]),
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
        "worker_init_fn": seed_worker,
    }
    if workers > 0:
        loader_common["prefetch_factor"] = 2
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        generator=generator,
        **loader_common,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_common)
    return train_loader, val_loader

