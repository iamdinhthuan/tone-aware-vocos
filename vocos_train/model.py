from __future__ import annotations

import torch
from torch import nn

from vocos.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from vocos.feature_extractors import MelSpectrogramFeatures
from vocos.heads import ISTFTHead
from vocos.loss import DiscriminatorLoss, FeatureMatchingLoss, GeneratorLoss, MelSpecReconstructionLoss
from vocos.models import VocosBackbone

from .tone_classifier import load_tone_classifier
from .tone_discriminator import CQTToneDiscriminator
from .tone_losses import ToneFeatureMatchingLoss, WeightedMultiResolutionSTFTLoss


class VocosGenerator(nn.Module):
    def __init__(self, data_config: dict, model_config: dict) -> None:
        super().__init__()
        self.feature_extractor = MelSpectrogramFeatures(
            sample_rate=data_config["sample_rate"],
            n_fft=model_config["n_fft"],
            hop_length=model_config["hop_length"],
            n_mels=model_config["n_mels"],
            padding="center",
        )
        self.backbone = VocosBackbone(
            input_channels=model_config["n_mels"],
            dim=model_config["dim"],
            intermediate_dim=model_config["intermediate_dim"],
            num_layers=model_config["num_layers"],
        )
        self.head = ISTFTHead(
            dim=model_config["dim"],
            n_fft=model_config["n_fft"],
            hop_length=model_config["hop_length"],
            padding="center",
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(audio)
        return self.head(self.backbone(features))


class VocosTrainingSystem(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self.tone_aware = config.get("tone_aware", {})
        self.generator = VocosGenerator(config["data"], config["model"])
        self.mpd = MultiPeriodDiscriminator()
        self.mrd = MultiResolutionDiscriminator()
        self.cqt_disc = (
            CQTToneDiscriminator(
                sample_rate=config["data"]["sample_rate"],
                fmin=float(self.tone_aware.get("cqt_fmin", 80.0)),
                n_bins=int(self.tone_aware.get("cqt_n_bins", 48)),
                bins_per_octave=int(self.tone_aware.get("cqt_bins_per_octave", 24)),
                hop_length=config["model"]["hop_length"],
            )
            if self.tone_aware.get("use_cqt_disc", False)
            else None
        )
        self.discriminator_loss = DiscriminatorLoss()
        self.generator_loss = GeneratorLoss()
        self.feature_matching_loss = FeatureMatchingLoss()
        self.mel_loss = MelSpecReconstructionLoss(
            sample_rate=config["data"]["sample_rate"],
            n_fft=config["model"]["n_fft"],
            hop_length=config["model"]["hop_length"],
            n_mels=config["model"]["n_mels"],
        )
        self.weighted_stft_loss = (
            WeightedMultiResolutionSTFTLoss(
                sample_rate=config["data"]["sample_rate"],
                fft_sizes=self.tone_aware.get("stft_fft_sizes", [512, 1024, 2048]),
                hop_length=config["model"]["hop_length"],
                low_hz=float(self.tone_aware.get("stft_low_hz", 1000.0)),
                w_low=float(self.tone_aware.get("stft_w_low", 3.0)),
            )
            if self.tone_aware.get("freq_weighted_stft", False)
            else None
        )
        self.tone_fm_loss = None
        if self.tone_aware.get("use_tone_loss", False):
            classifier = load_tone_classifier(self.tone_aware["tone_classifier_ckpt"])
            self.tone_fm_loss = ToneFeatureMatchingLoss(classifier)

    def discriminator_parameters(self) -> list[nn.Parameter]:
        params = list(self.mpd.parameters()) + list(self.mrd.parameters())
        if self.cqt_disc is not None:
            params += list(self.cqt_disc.parameters())
        return params

    def discriminator_objective(self, real: torch.Tensor, fake: torch.Tensor, mrd_coeff: float) -> dict:
        real_mp, fake_mp, _, _ = self.mpd(real, fake)
        real_mrd, fake_mrd, _, _ = self.mrd(real, fake)
        loss_mp, real_parts_mp, _ = self.discriminator_loss(real_mp, fake_mp)
        loss_mrd, real_parts_mrd, _ = self.discriminator_loss(real_mrd, fake_mrd)
        loss_mp = loss_mp / len(real_parts_mp)
        loss_mrd = loss_mrd / len(real_parts_mrd)
        zero = loss_mp.new_zeros(())
        loss_cqt = zero
        if self.cqt_disc is not None:
            real_cqt, fake_cqt, _, _ = self.cqt_disc(real, fake)
            loss_cqt, real_parts_cqt, _ = self.discriminator_loss(real_cqt, fake_cqt)
            loss_cqt = loss_cqt / len(real_parts_cqt)
        cqt_coeff = float(self.tone_aware.get("cqt_loss_coeff", mrd_coeff))
        return {
            "total": loss_mp + mrd_coeff * loss_mrd + cqt_coeff * loss_cqt,
            "mp": loss_mp,
            "mrd": loss_mrd,
            "cqt": loss_cqt,
        }

    def generator_objective(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
        mel_coeff: float,
        mrd_coeff: float,
        adversarial: bool,
    ) -> dict:
        mel = self.mel_loss(fake, real)
        zero = mel.new_zeros(())
        stft = zero
        tone = zero
        if self.weighted_stft_loss is not None:
            stft = self.weighted_stft_loss(fake, real)
        if self.tone_fm_loss is not None:
            tone = self.tone_fm_loss(real, fake)
        stft_coeff = float(self.tone_aware.get("stft_loss_coeff", 1.0))
        tone_coeff = float(self.tone_aware.get("tone_loss_coeff", 1.0))
        if not adversarial:
            total = mel_coeff * mel + stft_coeff * stft + tone_coeff * tone
            return {
                "total": total,
                "mel": mel,
                "mp": zero,
                "mrd": zero,
                "fm_mp": zero,
                "fm_mrd": zero,
                "cqt": zero,
                "fm_cqt": zero,
                "stft": stft,
                "tone": tone,
            }

        _, fake_mp, real_fmap_mp, fake_fmap_mp = self.mpd(real, fake)
        _, fake_mrd, real_fmap_mrd, fake_fmap_mrd = self.mrd(real, fake)
        gen_mp, parts_mp = self.generator_loss(fake_mp)
        gen_mrd, parts_mrd = self.generator_loss(fake_mrd)
        gen_mp = gen_mp / len(parts_mp)
        gen_mrd = gen_mrd / len(parts_mrd)
        fm_mp = self.feature_matching_loss(real_fmap_mp, fake_fmap_mp) / len(real_fmap_mp)
        fm_mrd = self.feature_matching_loss(real_fmap_mrd, fake_fmap_mrd) / len(real_fmap_mrd)
        gen_cqt = zero
        fm_cqt = zero
        if self.cqt_disc is not None:
            _, fake_cqt, real_fmap_cqt, fake_fmap_cqt = self.cqt_disc(real, fake)
            gen_cqt, parts_cqt = self.generator_loss(fake_cqt)
            gen_cqt = gen_cqt / len(parts_cqt)
            fm_cqt = self.feature_matching_loss(real_fmap_cqt, fake_fmap_cqt) / len(real_fmap_cqt)
        cqt_coeff = float(self.tone_aware.get("cqt_loss_coeff", mrd_coeff))
        cqt_fm_coeff = float(self.tone_aware.get("cqt_fm_coeff", cqt_coeff))
        total = (
            gen_mp
            + mrd_coeff * gen_mrd
            + cqt_coeff * gen_cqt
            + fm_mp
            + mrd_coeff * fm_mrd
            + cqt_fm_coeff * fm_cqt
            + mel_coeff * mel
            + stft_coeff * stft
            + tone_coeff * tone
        )
        return {
            "total": total,
            "mel": mel,
            "mp": gen_mp,
            "mrd": gen_mrd,
            "fm_mp": fm_mp,
            "fm_mrd": fm_mrd,
            "cqt": gen_cqt,
            "fm_cqt": fm_cqt,
            "stft": stft,
            "tone": tone,
        }


def set_discriminator_grad(system: VocosTrainingSystem, enabled: bool) -> None:
    modules = [system.mpd, system.mrd]
    if system.cqt_disc is not None:
        modules.append(system.cqt_disc)
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)
