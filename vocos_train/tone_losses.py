from __future__ import annotations

import torch
from torch import nn

from .tone_classifier import ToneClassifier


class WeightedMultiResolutionSTFTLoss(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        fft_sizes: list[int] | tuple[int, ...] = (512, 1024, 2048),
        hop_length: int = 256,
        low_hz: float = 1000.0,
        w_low: float = 3.0,
        w_high: float = 1.0,
    ) -> None:
        super().__init__()
        self.fft_sizes = [int(value) for value in fft_sizes]
        self.hop_length = int(hop_length)
        self.register_buffer("_dummy", torch.empty(0), persistent=False)
        for n_fft in self.fft_sizes:
            freqs = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1)
            weight = torch.where(freqs <= low_hz, torch.full_like(freqs, w_low), torch.full_like(freqs, w_high))
            self.register_buffer(f"weight_{n_fft}", weight.view(1, -1, 1), persistent=False)

    def _mag(self, audio: torch.Tensor, n_fft: int) -> torch.Tensor:
        audio = audio.float()
        window = torch.hann_window(n_fft, device=audio.device, dtype=audio.dtype)
        spec = torch.stft(
            audio,
            n_fft=n_fft,
            hop_length=self.hop_length,
            win_length=n_fft,
            window=window,
            center=True,
            return_complex=True,
        )
        return spec.abs().clamp_min(1e-7)

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        total = real.new_zeros((), dtype=torch.float32)
        for n_fft in self.fft_sizes:
            fake_mag = self._mag(fake, n_fft)
            real_mag = self._mag(real, n_fft)
            weight = getattr(self, f"weight_{n_fft}").to(device=real_mag.device, dtype=real_mag.dtype)
            diff = (fake_mag - real_mag).abs() * weight
            sc = torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(real_mag * weight).clamp_min(1e-7)
            log_mag = (torch.log(fake_mag) - torch.log(real_mag)).abs().mul(weight).mean()
            total = total + sc + log_mag
        return total / len(self.fft_sizes)


class ToneFeatureMatchingLoss(nn.Module):
    def __init__(self, classifier: ToneClassifier) -> None:
        super().__init__()
        self.classifier = classifier
        for parameter in self.classifier.parameters():
            parameter.requires_grad_(False)
        self.classifier.eval()

    def forward(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        self.classifier.eval()
        with torch.no_grad():
            _, feats_real = self.classifier(real.float(), return_features=True)
        _, feats_fake = self.classifier(fake.float(), return_features=True)
        total = fake.new_zeros((), dtype=torch.float32)
        for real_feat, fake_feat in zip(feats_real, feats_fake):
            total = total + (real_feat.detach() - fake_feat).abs().mean()
        return total / len(feats_real)
