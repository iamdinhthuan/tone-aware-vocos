from __future__ import annotations

import argparse
import hashlib
import os
import random
from pathlib import Path

from tqdm import tqdm


def voice_slug(filename: str) -> str:
    stem = Path(filename).stem
    if "_" not in stem:
        raise ValueError("filename does not contain the expected '<id>_<voice>' separator")
    return stem.split("_", 1)[1]


def is_validation_voice(slug: str, seed: int, val_percent: float) -> bool:
    digest = hashlib.blake2b(f"{seed}:{slug}".encode("utf-8"), digest_size=8).digest()
    bucket = int.from_bytes(digest, "big") / float(2**64)
    return bucket < val_percent / 100.0


def iter_mp3(root: Path, recursive: bool):
    if recursive:
        for directory, _, files in os.walk(root):
            base = Path(directory)
            for filename in files:
                if filename.lower().endswith(".mp3"):
                    yield base / filename
    else:
        with os.scandir(root) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.lower().endswith(".mp3"):
                    yield Path(entry.path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create leakage-safe Vocos manifests from MP3 files")
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--output-dir", default="manifests")
    parser.add_argument("--val-percent", type=float, default=1.0, help="Percentage of voice identities reserved for validation")
    parser.add_argument("--max-val-files", type=int, default=10000, help="Reservoir-sample validation files; 0 keeps all")
    parser.add_argument("--seed", type=int, default=4444)
    parser.add_argument("--min-bytes", type=int, default=1024)
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.val_percent < 100.0:
        parser.error("--val-percent must be between 0 and 100")
    root = Path(args.audio_root).expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Audio root does not exist: {root}")
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    train_tmp, val_tmp = output / "train.txt.tmp", output / "val.txt.tmp"
    rng = random.Random(args.seed)
    val_reservoir: list[str] = []
    train_count = val_seen = malformed = too_small = 0
    train_voices: set[str] = set()
    val_voices: set[str] = set()

    with train_tmp.open("w", encoding="utf-8", buffering=1024 * 1024) as train_handle:
        for path in tqdm(iter_mp3(root, args.recursive), unit="file", dynamic_ncols=True):
            try:
                if path.stat().st_size < args.min_bytes:
                    too_small += 1
                    continue
                slug = voice_slug(path.name)
            except (OSError, ValueError):
                malformed += 1
                continue
            relative = path.relative_to(root).as_posix()
            if is_validation_voice(slug, args.seed, args.val_percent):
                val_voices.add(slug)
                val_seen += 1
                if args.max_val_files <= 0 or len(val_reservoir) < args.max_val_files:
                    val_reservoir.append(relative)
                else:
                    replacement = rng.randrange(val_seen)
                    if replacement < args.max_val_files:
                        val_reservoir[replacement] = relative
            else:
                train_voices.add(slug)
                train_handle.write(relative + "\n")
                train_count += 1

    rng.shuffle(val_reservoir)
    with val_tmp.open("w", encoding="utf-8", buffering=1024 * 1024) as val_handle:
        val_handle.writelines(path + "\n" for path in val_reservoir)
    os.replace(train_tmp, output / "train.txt")
    os.replace(val_tmp, output / "val.txt")
    print(
        f"train_files={train_count:,} train_voices={len(train_voices):,} "
        f"val_files={len(val_reservoir):,}/{val_seen:,} val_voices={len(val_voices):,} "
        f"malformed={malformed:,} too_small={too_small:,}"
    )


if __name__ == "__main__":
    main()

