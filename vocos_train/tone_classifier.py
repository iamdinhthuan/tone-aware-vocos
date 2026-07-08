from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import torchaudio


class ToneClassifier(nn.Module):
    def __init__(self, sample_rate: int = 24000, n_mels: int = 80, n_tones: int = 6, hidden: int = 128) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_tones = n_tones
        self.hidden = hidden
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=n_mels,
            f_min=40,
            f_max=sample_rate // 2,
            power=1.0,
        )
        self.conv = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU()),
                nn.Sequential(nn.Conv2d(32, 64, 3, stride=(2, 1), padding=1), nn.BatchNorm2d(64), nn.GELU()),
                nn.Sequential(nn.Conv2d(64, 128, 3, stride=(2, 1), padding=1), nn.BatchNorm2d(128), nn.GELU()),
            ]
        )
        self.gru = nn.GRU(128 * (n_mels // 4), hidden, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * hidden, n_tones)

    def forward(self, wav: torch.Tensor, return_features: bool = False):
        x = torch.log(self.melspec(wav.float()).clamp_min(1e-5)).unsqueeze(1)
        feats = []
        for layer in self.conv:
            x = layer(x)
            feats.append(x)
        batch, channels, mel, frames = x.shape
        seq = x.permute(0, 3, 1, 2).reshape(batch, frames, channels * mel)
        out, _ = self.gru(seq)
        logits = self.head(out.mean(dim=1))
        if return_features:
            return logits, feats
        return logits

    def config(self) -> dict:
        return {
            "sample_rate": self.sample_rate,
            "n_mels": self.n_mels,
            "n_tones": self.n_tones,
            "hidden": self.hidden,
        }


def load_tone_classifier(path: str | Path, device: torch.device | str = "cpu") -> ToneClassifier:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("config", {})
    model = ToneClassifier(
        sample_rate=int(cfg.get("sample_rate", 24000)),
        n_mels=int(cfg.get("n_mels", 80)),
        n_tones=int(cfg.get("n_tones", 6)),
        hidden=int(cfg.get("hidden", 128)),
    )
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state)
    return model.to(device)
