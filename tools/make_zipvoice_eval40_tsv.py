from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CUSTOM_DEV = Path("/data_nvme/ZipVoice/egs/zipvoice/data/raw/custom_dev.tsv")
TARGETS = ROOT / "human_eval_gradio" / "target_sentences_40.txt"
OUT = ROOT / "human_eval_gradio" / "zipvoice_eval40.tsv"
OUT_META = ROOT / "human_eval_gradio" / "zipvoice_eval40_metadata.csv"


def read_targets() -> list[str]:
    targets = [line.strip() for line in TARGETS.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(targets) != 40:
        raise ValueError(f"Expected exactly 40 target sentences, got {len(targets)}")
    return targets


def read_prompts(n: int) -> list[tuple[str, str, str]]:
    prompts: list[tuple[str, str, str]] = []
    with CUSTOM_DEV.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            _, prompt_text, prompt_wav = parts
            if not Path(prompt_wav).is_file():
                continue
            # Keep prompts not too long for fast, stable zero-shot synthesis.
            if 20 <= len(prompt_text) <= 180:
                prompts.append((prompt_text, prompt_wav, parts[0]))
            if len(prompts) >= n:
                break
    if len(prompts) < n:
        raise RuntimeError(f"Only found {len(prompts)} usable prompts in {CUSTOM_DEV}")
    return prompts


def main() -> None:
    targets = read_targets()
    prompts = read_prompts(len(targets))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as f_tsv, OUT_META.open("w", encoding="utf-8", newline="") as f_meta:
        meta_writer = csv.DictWriter(
            f_meta,
            fieldnames=["wav_name", "prompt_id", "prompt_text", "prompt_wav", "target_text"],
        )
        meta_writer.writeheader()
        for idx, (target, (prompt_text, prompt_wav, prompt_id)) in enumerate(zip(targets, prompts), start=1):
            wav_name = f"sample_{idx:03d}"
            f_tsv.write(f"{wav_name}\t{prompt_text}\t{prompt_wav}\t{target}\n")
            meta_writer.writerow(
                {
                    "wav_name": wav_name,
                    "prompt_id": prompt_id,
                    "prompt_text": prompt_text,
                    "prompt_wav": prompt_wav,
                    "target_text": target,
                }
            )
    print(OUT)
    print(OUT_META)


if __name__ == "__main__":
    main()
