from __future__ import annotations

import argparse
import csv
from pathlib import Path

from tqdm import tqdm

from vocos_train.vietnamese_tones import tone_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract syllable-level tone labels from MFA TextGrid outputs")
    parser.add_argument("--mfa-manifest", required=True, help="TSV written by prepare_mfa_corpus.py")
    parser.add_argument("--textgrid-dir", required=True)
    parser.add_argument("--output", default="manifests/tone_segments.tsv")
    parser.add_argument("--min-duration", type=float, default=0.08)
    parser.add_argument("--max-duration", type=float, default=1.2)
    parser.add_argument("--tier-name", default="words")
    return parser.parse_args()


def load_intervals(textgrid_path: Path, tier_name: str):
    try:
        from praatio import textgrid
    except ImportError as exc:
        raise ImportError("extract_tone_segments.py requires praatio. Install with: pip install praatio") from exc
    tg = textgrid.openTextgrid(str(textgrid_path), includeEmptyIntervals=False)
    tier = tg.getTier(tier_name) if tier_name in tg.tierNames else tg.getTier(tg.tierNames[0])
    for entry in tier.entries:
        label = entry.label.strip()
        if label:
            yield float(entry.start), float(entry.end), label


def main() -> None:
    args = parse_args()
    textgrid_dir = Path(args.textgrid_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    kept = skipped = 0
    with open(args.mfa_manifest, "r", encoding="utf-8") as handle, output.open("w", encoding="utf-8") as out:
        reader = csv.DictReader(handle, delimiter="\t")
        writer = csv.writer(out, delimiter="\t")
        writer.writerow(["audio", "start", "end", "tone", "syllable", "speaker"])
        for row in tqdm(reader, desc="tone-segments"):
            tg_path = textgrid_dir / (Path(row["audio"]).stem + ".TextGrid")
            if not tg_path.is_file():
                skipped += 1
                continue
            for start, end, label in load_intervals(tg_path, args.tier_name):
                duration = end - start
                if duration < args.min_duration or duration > args.max_duration:
                    continue
                writer.writerow([row["audio"], f"{start:.6f}", f"{end:.6f}", tone_id(label), label, row["speaker"]])
                kept += 1
    print(f"kept={kept} skipped_textgrids={skipped} output={output}")


if __name__ == "__main__":
    main()
