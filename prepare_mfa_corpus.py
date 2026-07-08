from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import subprocess
from pathlib import Path

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a small MFA corpus from metadata.txt without expanding the full MP3 dataset")
    parser.add_argument("--metadata", default="/media/huy/data1tb/huy/data_vox/metadata.txt")
    parser.add_argument("--data-root", default="/media/huy/data1tb/huy/data_vox")
    parser.add_argument("--output", default="mfa_corpus/vi_shard")
    parser.add_argument("--max-files", type=int, default=100000)
    parser.add_argument("--speaker-filter", help="Optional substring filter on speaker name")
    parser.add_argument("--include-list", help="Optional newline list of relative audio paths to include")
    parser.add_argument("--format", choices=["flac", "wav"], default="flac")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_stem(rel_audio: str) -> str:
    digest = hashlib.sha1(rel_audio.encode("utf-8")).hexdigest()[:10]
    return f"{Path(rel_audio).stem}_{digest}"


def convert_audio(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-ac", "1", "-ar", "24000", str(dst)],
        check=True,
    )


def main() -> None:
    args = parse_args()
    metadata = Path(args.metadata)
    data_root = Path(args.data_root)
    output = Path(args.output)
    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "mfa_manifest.tsv"
    include = None
    if args.include_list:
        include_path = Path(args.include_list)
        include = set()
        with include_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                value = line.strip()
                if not value:
                    continue
                include.add(value)
                include.add(f"audio/{value}" if not value.startswith("audio/") else value.removeprefix("audio/"))

    written = 0
    with metadata.open("r", encoding="utf-8", errors="replace") as handle, manifest_path.open("w", encoding="utf-8") as out:
        writer = csv.writer(out, delimiter="\t")
        writer.writerow(["audio", "lab", "source_audio", "text", "speaker"])
        for line in tqdm(handle, desc="mfa-corpus"):
            parts = line.rstrip("\n").split("|")
            if len(parts) < 3:
                continue
            rel_audio, text, speaker = parts[0], parts[1], parts[2]
            if include is not None and rel_audio not in include:
                continue
            if args.speaker_filter and args.speaker_filter.lower() not in speaker.lower():
                continue
            src = data_root / rel_audio
            if not src.is_file():
                continue
            stem = safe_stem(rel_audio)
            audio_dst = output / f"{stem}.{args.format}"
            lab_dst = output / f"{stem}.lab"
            convert_audio(src, audio_dst)
            lab_dst.write_text(text.strip() + "\n", encoding="utf-8")
            writer.writerow([audio_dst, lab_dst, src, text, speaker])
            written += 1
            if written >= args.max_files:
                break
    print(f"written={written} output={output} manifest={manifest_path}")


if __name__ == "__main__":
    main()
