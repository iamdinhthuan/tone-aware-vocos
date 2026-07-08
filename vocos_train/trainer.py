from __future__ import annotations

import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .config import validate_config
from .data import create_dataloaders
from .model import VocosTrainingSystem, set_discriminator_grad


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_schedule(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    def factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))

    return LambdaLR(optimizer, factor)


def infinite_batches(loader) -> Iterator[torch.Tensor]:
    while True:
        yield from loader


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def optimizer_step(
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    parameters,
    scaler: torch.amp.GradScaler | None,
    grad_clip_norm: float,
) -> float:
    if scaler is None:
        loss.backward()
        grad_norm = clip_grad_norm_(parameters, grad_clip_norm)
        optimizer.step()
    else:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = clip_grad_norm_(parameters, grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
    return float(grad_norm)


@torch.inference_mode()
def validate(
    system: VocosTrainingSystem,
    val_loader,
    device: torch.device,
    precision: str,
    max_batches: int,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    system.eval()
    total = 0.0
    count = 0
    example_real = example_fake = None
    for batch_index, real in enumerate(val_loader):
        if batch_index >= max_batches:
            break
        real = real.to(device, non_blocking=True)
        with autocast_context(device, precision):
            fake = system.generator(real)
            loss = system.mel_loss(fake.float(), real.float())
        total += float(loss) * real.shape[0]
        count += real.shape[0]
        if example_real is None:
            example_real = real[0].detach().float().cpu()
            example_fake = fake[0].detach().float().cpu()
    system.train()
    if count == 0 or example_real is None or example_fake is None:
        raise RuntimeError("Validation loader produced no batches")
    return total / count, example_real, example_fake


def checkpoint_state(
    system: VocosTrainingSystem,
    optimizer_g,
    optimizer_d,
    scheduler_g,
    scheduler_d,
    scaler,
    step: int,
    best_val: float,
    config: dict,
) -> dict:
    return {
        "format_version": 1,
        "step": step,
        "best_val": best_val,
        "generator": system.generator.state_dict(),
        "mpd": system.mpd.state_dict(),
        "mrd": system.mrd.state_dict(),
        "cqt_disc": system.cqt_disc.state_dict() if system.cqt_disc is not None else None,
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
        "scheduler_g": scheduler_g.state_dict(),
        "scheduler_d": scheduler_d.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
    }


def atomic_save(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def atomic_symlink(target_name: str, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(target_name)
    os.replace(temporary, path)


def inference_state(system: VocosTrainingSystem, step: int, val_loss: float, config: dict) -> dict:
    return {
        "format_version": 1,
        "step": step,
        "val_loss": val_loss,
        "generator": system.generator.state_dict(),
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
    }


def best_checkpoint_loss(path: Path) -> float:
    try:
        return float(path.stem.rsplit("_val_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid best-checkpoint filename: {path.name}") from exc


def save_top_k_best(
    system: VocosTrainingSystem,
    output_dir: Path,
    step: int,
    val_loss: float,
    config: dict,
    top_k: int,
) -> list[Path]:
    existing = list(output_dir.glob("best_step_*_val_*.pt"))
    should_save = len(existing) < top_k or val_loss < max(best_checkpoint_loss(path) for path in existing)
    if should_save:
        candidate = output_dir / f"best_step_{step}_val_{val_loss:.6f}.pt"
        atomic_save(inference_state(system, step, val_loss, config), candidate)
        existing.append(candidate)

    ranked = sorted(existing, key=best_checkpoint_loss)
    for path in ranked[top_k:]:
        path.unlink(missing_ok=True)
    ranked = ranked[:top_k]
    if ranked:
        atomic_symlink(ranked[0].name, output_dir / "best.pt")
    return ranked


def prune_checkpoints(directory: Path, keep: int) -> None:
    checkpoints = sorted(directory.glob("step_*.pt"), key=lambda path: int(path.stem.split("_")[1]))
    for path in checkpoints[:-keep] if keep > 0 else checkpoints:
        path.unlink(missing_ok=True)


def train(config: dict, resume: str | None = None, device_name: str | None = None) -> None:
    validate_config(config)
    seed_everything(int(config["seed"]))
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    data_cfg, train_cfg = config["data"], config["training"]
    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(train_cfg["log_dir"]).mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump({k: v for k, v in config.items() if not k.startswith("_")}, handle, sort_keys=False)

    train_loader, val_loader = create_dataloaders(config, device)
    system = VocosTrainingSystem(config).to(device)
    generator_parameters = list(system.generator.parameters())
    discriminator_parameters = system.discriminator_parameters()
    betas = tuple(float(value) for value in train_cfg["betas"])
    optimizer_g = torch.optim.AdamW(generator_parameters, lr=train_cfg["learning_rate"], betas=betas)
    optimizer_d = torch.optim.AdamW(discriminator_parameters, lr=train_cfg["learning_rate"], betas=betas)
    scheduler_g = cosine_schedule(optimizer_g, train_cfg["warmup_steps"], train_cfg["max_steps"])
    adversarial_steps = max(1, train_cfg["max_steps"] - train_cfg["pretrain_mel_steps"])
    scheduler_d = cosine_schedule(optimizer_d, train_cfg["warmup_steps"], adversarial_steps)
    use_scaler = device.type == "cuda" and train_cfg["precision"] == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_scaler else None
    start_step, best_val = 0, float("inf")

    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        if "optimizer_g" not in checkpoint:
            raise ValueError(f"Checkpoint is inference-only and cannot resume training: {resume}")
        system.generator.load_state_dict(checkpoint["generator"])
        system.mpd.load_state_dict(checkpoint["mpd"])
        system.mrd.load_state_dict(checkpoint["mrd"])
        if system.cqt_disc is not None:
            if checkpoint.get("cqt_disc") is None:
                raise ValueError("Cannot resume tone-aware CQT run from checkpoint without cqt_disc state")
            system.cqt_disc.load_state_dict(checkpoint["cqt_disc"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        scheduler_g.load_state_dict(checkpoint["scheduler_g"])
        scheduler_d.load_state_dict(checkpoint["scheduler_d"])
        if scaler is not None and checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_step = int(checkpoint["step"])
        best_val = float(checkpoint.get("best_val", best_val))

    writer = SummaryWriter(train_cfg["log_dir"], purge_step=start_step or None)
    total_parameters = sum(parameter.numel() for parameter in system.parameters())
    print(f"device={device} parameters={total_parameters / 1e6:.2f}M train_files={len(train_loader.dataset):,} val_files={len(val_loader.dataset):,}")
    if resume:
        print(f"resumed={resume} step={start_step:,}")

    batches = infinite_batches(train_loader)
    progress = tqdm(range(start_step, train_cfg["max_steps"]), initial=start_step, total=train_cfg["max_steps"], dynamic_ncols=True)
    last_log_time = time.perf_counter()
    running: dict[str, float] = {}
    system.train()

    try:
        for index in progress:
            step = index + 1
            real = next(batches).to(device, non_blocking=True)
            adversarial = step > train_cfg["pretrain_mel_steps"]

            if adversarial:
                set_discriminator_grad(system, True)
                optimizer_d.zero_grad(set_to_none=True)
                with torch.no_grad(), autocast_context(device, train_cfg["precision"]):
                    fake_for_d = system.generator(real)
                with autocast_context(device, train_cfg["precision"]):
                    d_losses = system.discriminator_objective(real, fake_for_d.detach(), train_cfg["mrd_loss_coeff"])
                d_grad = optimizer_step(
                    d_losses["total"], optimizer_d, discriminator_parameters, scaler, train_cfg["grad_clip_norm"]
                )
                scheduler_d.step()
            else:
                d_losses = {
                    "total": real.new_zeros(()),
                    "mp": real.new_zeros(()),
                    "mrd": real.new_zeros(()),
                    "cqt": real.new_zeros(()),
                }
                d_grad = 0.0

            set_discriminator_grad(system, False)
            optimizer_g.zero_grad(set_to_none=True)
            with autocast_context(device, train_cfg["precision"]):
                fake = system.generator(real)
                g_losses = system.generator_objective(
                    real,
                    fake,
                    train_cfg["mel_loss_coeff"],
                    train_cfg["mrd_loss_coeff"],
                    adversarial,
                )
            g_grad = optimizer_step(
                g_losses["total"], optimizer_g, generator_parameters, scaler, train_cfg["grad_clip_norm"]
            )
            scheduler_g.step()
            set_discriminator_grad(system, True)

            metrics = {
                "g_total": float(g_losses["total"].detach()),
                "mel": float(g_losses["mel"].detach()),
                "g_mp": float(g_losses["mp"].detach()),
                "g_mrd": float(g_losses["mrd"].detach()),
                "g_cqt": float(g_losses["cqt"].detach()),
                "fm_mp": float(g_losses["fm_mp"].detach()),
                "fm_mrd": float(g_losses["fm_mrd"].detach()),
                "fm_cqt": float(g_losses["fm_cqt"].detach()),
                "stft": float(g_losses["stft"].detach()),
                "tone": float(g_losses["tone"].detach()),
                "d_total": float(d_losses["total"].detach()),
                "d_cqt": float(d_losses["cqt"].detach()),
                "g_grad": g_grad,
                "d_grad": d_grad,
            }
            for key, value in metrics.items():
                running[key] = running.get(key, 0.0) + value

            if step % train_cfg["log_every_steps"] == 0:
                elapsed = time.perf_counter() - last_log_time
                count = train_cfg["log_every_steps"]
                averaged = {key: value / count for key, value in running.items()}
                for key, value in averaged.items():
                    writer.add_scalar(f"train/{key}", value, step)
                writer.add_scalar("train/lr_g", scheduler_g.get_last_lr()[0], step)
                writer.add_scalar("train/lr_d", scheduler_d.get_last_lr()[0], step)
                writer.add_scalar("train/steps_per_second", count / elapsed, step)
                progress.set_postfix(mel=f"{averaged['mel']:.4f}", g=f"{averaged['g_total']:.2f}", d=f"{averaged['d_total']:.2f}")
                running.clear()
                last_log_time = time.perf_counter()

            if step % train_cfg["audio_every_steps"] == 0:
                writer.add_audio("train/real", real[0].detach().float().cpu(), step, data_cfg["sample_rate"])
                writer.add_audio("train/generated", fake[0].detach().float().cpu().clamp(-1, 1), step, data_cfg["sample_rate"])

            validation_due = step % train_cfg["validate_every_steps"] == 0 or step == train_cfg["max_steps"]
            if validation_due:
                val_loss, val_real, val_fake = validate(
                    system, val_loader, device, train_cfg["precision"], train_cfg["val_batches"]
                )
                writer.add_scalar("validation/mel_loss", val_loss, step)
                writer.add_audio("validation/real", val_real, step, data_cfg["sample_rate"])
                writer.add_audio("validation/generated", val_fake.clamp(-1, 1), step, data_cfg["sample_rate"])
                print(f"\nstep={step:,} validation_mel={val_loss:.6f}")
                ranked_best = save_top_k_best(
                    system,
                    output_dir,
                    step,
                    val_loss,
                    config,
                    int(train_cfg.get("save_top_k_best", 3)),
                )
                best_val = best_checkpoint_loss(ranked_best[0])
                print("top_val=" + ", ".join(f"{best_checkpoint_loss(path):.6f}" for path in ranked_best))

            checkpoint_due = step % train_cfg["checkpoint_every_steps"] == 0 or step == train_cfg["max_steps"]
            if checkpoint_due:
                state = checkpoint_state(system, optimizer_g, optimizer_d, scheduler_g, scheduler_d, scaler, step, best_val, config)
                step_path = output_dir / f"step_{step}.pt"
                atomic_save(state, step_path)
                atomic_symlink(step_path.name, output_dir / "last.pt")
                prune_checkpoints(output_dir, int(train_cfg["keep_last_checkpoints"]))
                writer.flush()
    except KeyboardInterrupt:
        state = checkpoint_state(system, optimizer_g, optimizer_d, scheduler_g, scheduler_d, scaler, step, best_val, config)
        atomic_save(state, output_dir / "interrupted.pt")
        print(f"\nInterrupted checkpoint saved at step {step:,}")
        raise
    finally:
        writer.close()
