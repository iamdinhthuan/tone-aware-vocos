from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils import weight_norm


class CQTToneDiscriminator(nn.Module):
    def __init__(
        self,
        sample_rate: int = 24000,
        fmin: float = 80.0,
        n_bins: int = 48,
        bins_per_octave: int = 24,
        hop_length: int = 256,
    ) -> None:
        super().__init__()
        try:
            from nnAudio.features import CQT1992v2
        except ImportError as exc:
            raise ImportError("CQTToneDiscriminator requires nnAudio. Install with: pip install nnAudio") from exc
        self.cqt = CQT1992v2(
            sr=sample_rate,
            fmin=fmin,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            hop_length=hop_length,
            output_format="Magnitude",
        )
        channels = [1, 32, 64, 128, 128]
        self.convs = nn.ModuleList(
            [
                weight_norm(nn.Conv2d(channels[i], channels[i + 1], (3, 3), (1, 1), padding=(1, 1)))
                for i in range(len(channels) - 1)
            ]
        )
        self.act = nn.LeakyReLU(0.1)
        self.post = weight_norm(nn.Conv2d(128, 1, (3, 1), (1, 1), padding=(1, 0)))

    def forward_one(self, audio: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        z = torch.log(self.cqt(audio.float()).clamp_min(1e-5)).unsqueeze(1)
        fmap = []
        for conv in self.convs:
            z = self.act(conv(z))
            fmap.append(z)
        score = self.post(z)
        return score, fmap

    def forward(self, real: torch.Tensor, fake: torch.Tensor):
        real_score, real_fmap = self.forward_one(real)
        fake_score, fake_fmap = self.forward_one(fake)
        return [real_score], [fake_score], [real_fmap], [fake_fmap]
