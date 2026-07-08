from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from vocos_train.tone_classifier import ToneClassifier


class ToneSegmentDataset(Dataset):
    def __init__(self, manifest: str, sample_rate: int, num_samples: int) -> None:
        self.rows = []
        with open(manifest, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                self.rows.append(row)
        if not self.rows:
            raise ValueError(f"Empty tone manifest: {manifest}")
        self.sample_rate = sample_rate
        self.num_samples = num_samples

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = row["audio"]
        start = float(row["start"])
        end = float(row["end"])
        info = torchaudio.info(path)
        frame_offset = max(0, int(start * info.sample_rate))
        num_frames = max(1, int((end - start) * info.sample_rate))
        wav, sr = torchaudio.load(path, frame_offset=frame_offset, num_frames=num_frames)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        wav = wav[0].float()
        peak = wav.abs().amax().clamp_min(1e-7)
        wav = wav * ((10 ** (random.uniform(-6.0, -1.0) / 20.0)) / peak)
        if wav.numel() < self.num_samples:
            wav = torch.nn.functional.pad(wav, (0, self.num_samples - wav.numel()))
        elif wav.numel() > self.num_samples:
            start_idx = random.randint(0, wav.numel() - self.num_samples)
            wav = wav[start_idx : start_idx + self.num_samples]
        return wav.contiguous(), torch.tensor(int(row["tone"]), dtype=torch.long)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train syllable-level Vietnamese tone classifier")
    parser.add_argument("--manifest", default="manifests/tone_segments.tsv")
    parser.add_argument("--output", default="checkpoints/tone_classifier/main.pt")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--num-samples", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=4444)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for wav, tone in loader:
            wav, tone = wav.to(device), tone.to(device)
            pred = model(wav).argmax(dim=1)
            correct += int((pred == tone).sum())
            total += int(tone.numel())
    model.train()
    return correct / max(1, total)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    dataset = ToneSegmentDataset(args.manifest, args.sample_rate, args.num_samples)
    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device(args.device)
    model = ToneClassifier(sample_rate=args.sample_rate).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    best = 0.0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        losses = []
        for wav, tone in tqdm(train_loader, desc=f"epoch {epoch}"):
            wav, tone = wav.to(device, non_blocking=True), tone.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(wav), tone)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        acc = accuracy(model, val_loader, device)
        print(f"epoch={epoch} train_loss={sum(losses)/len(losses):.4f} val_acc={acc:.4f}")
        if acc >= best:
            best = acc
            torch.save({"model": model.state_dict(), "config": model.config(), "val_acc": best, "epoch": epoch}, output)
            print(f"saved={output} val_acc={best:.4f}")


if __name__ == "__main__":
    main()
