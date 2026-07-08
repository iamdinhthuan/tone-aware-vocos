from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import sys
from pathlib import Path

import torch
from lhotse.utils import fix_random_seed


ZIPVOICE_ROOT = Path("/data_nvme/ZipVoice")
if str(ZIPVOICE_ROOT) not in sys.path:
    sys.path.insert(0, str(ZIPVOICE_ROOT))

from zipvoice.bin.infer_zipvoice import (  # noqa: E402
    VocosFbank,
    ZipVoice,
    get_vocoder,
    generate_sentence,
    load_checkpoint,
)
from zipvoice.tokenizer.tokenizer import SimpleTokenizer  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = Path("/data_nvme/ZipVoice/egs/zipvoice/exp/zipvoice_vi_simple_finetune")
DEFAULT_TEST_LIST = ROOT / "human_eval_gradio" / "zipvoice_eval40.tsv"
DEFAULT_OUT_ROOT = ROOT / "zipvoice_vocoder_samples_eval40"
DEFAULT_BASELINE_VOCODER = ROOT / "exported_vocoders" / "baseline"
DEFAULT_FINETUNED_VOCODER = ROOT / "exported_vocoders" / "plus_cb"


def read_test_list(path: Path) -> list[dict[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                raise ValueError(f"Invalid test-list line {line_no}: expected 4 tab-separated fields")
            wav_name, prompt_text, prompt_wav, text = parts
            rows.append(
                {
                    "wav_name": wav_name,
                    "prompt_text": prompt_text,
                    "prompt_wav": prompt_wav,
                    "text": text,
                }
            )
    return rows


def load_zipvoice(model_dir: Path, checkpoint_name: str, device: torch.device) -> tuple[ZipVoice, SimpleTokenizer, VocosFbank, int]:
    with (model_dir / "model.json").open("r", encoding="utf-8") as handle:
        model_config = json.load(handle)
    token_file = model_dir / "tokens.txt"
    tokenizer = SimpleTokenizer(token_file=token_file)
    tokenizer_config = {"vocab_size": tokenizer.vocab_size, "pad_id": tokenizer.pad_id}
    model = ZipVoice(**model_config["model"], **tokenizer_config)
    load_checkpoint(filename=model_dir / checkpoint_name, model=model, strict=True)
    model = model.to(device).eval()
    feature_extractor = VocosFbank()
    sampling_rate = int(model_config["feature"]["sampling_rate"])
    return model, tokenizer, feature_extractor, sampling_rate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate matched ZipVoice samples decoded by baseline and fine-tuned vocoders"
    )
    parser.add_argument("--test-list", type=Path, default=DEFAULT_TEST_LIST)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--checkpoint-name", default="best-valid-loss.pt")
    parser.add_argument("--baseline-vocoder", type=Path, default=DEFAULT_BASELINE_VOCODER)
    parser.add_argument("--finetuned-vocoder", type=Path, default=DEFAULT_FINETUNED_VOCODER)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--num-step", type=int, default=16)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--t-shift", type=float, default=0.5)
    parser.add_argument("--target-rms", type=float, default=0.1)
    parser.add_argument("--feat-scale", type=float, default=0.1)
    parser.add_argument("--max-duration", type=float, default=100.0)
    parser.add_argument("--remove-long-sil", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
        force=True,
    )
    fix_random_seed(args.seed)
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Device: %s", device)

    baseline_dir = args.out_root / "baseline"
    finetuned_dir = args.out_root / "plus_cb"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    finetuned_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.out_root / "generation_metadata.csv"
    rows = read_test_list(args.test_list)

    model, tokenizer, feature_extractor, sampling_rate = load_zipvoice(
        args.model_dir, args.checkpoint_name, device
    )
    baseline_vocoder = get_vocoder(str(args.baseline_vocoder)).to(device).eval()
    finetuned_vocoder = get_vocoder(str(args.finetuned_vocoder)).to(device).eval()

    start_all = dt.datetime.now()
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "wav_name",
                "baseline_path",
                "finetuned_plus_cb_path",
                "prompt_text",
                "prompt_wav",
                "target_text",
                "rtf",
                "rtf_no_vocoder",
                "rtf_vocoder_baseline_primary",
                "wav_seconds",
            ],
        )
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            wav_name = row["wav_name"]
            baseline_path = baseline_dir / f"{wav_name}.wav"
            finetuned_path = finetuned_dir / f"{wav_name}.wav"
            if baseline_path.exists() and finetuned_path.exists() and not args.overwrite:
                logging.info("[%03d/%03d] Skip existing %s", index, len(rows), wav_name)
                continue
            logging.info("[%03d/%03d] Generate %s", index, len(rows), wav_name)
            metrics = generate_sentence(
                save_path=str(baseline_path),
                secondary_save_path=str(finetuned_path),
                prompt_text=row["prompt_text"],
                prompt_wav=row["prompt_wav"],
                text=row["text"],
                model=model,
                vocoder=baseline_vocoder,
                secondary_vocoder=finetuned_vocoder,
                tokenizer=tokenizer,
                feature_extractor=feature_extractor,
                device=device,
                num_step=args.num_step,
                guidance_scale=args.guidance_scale,
                speed=args.speed,
                t_shift=args.t_shift,
                target_rms=args.target_rms,
                feat_scale=args.feat_scale,
                sampling_rate=sampling_rate,
                max_duration=args.max_duration,
                remove_long_sil=args.remove_long_sil,
            )
            writer.writerow(
                {
                    "wav_name": wav_name,
                    "baseline_path": baseline_path,
                    "finetuned_plus_cb_path": finetuned_path,
                    "prompt_text": row["prompt_text"],
                    "prompt_wav": row["prompt_wav"],
                    "target_text": row["text"],
                    "rtf": f"{metrics['rtf']:.6f}",
                    "rtf_no_vocoder": f"{metrics['rtf_no_vocoder']:.6f}",
                    "rtf_vocoder_baseline_primary": f"{metrics['rtf_vocoder']:.6f}",
                    "wav_seconds": f"{metrics['wav_seconds']:.6f}",
                }
            )
            handle.flush()
    elapsed = (dt.datetime.now() - start_all).total_seconds()
    logging.info("Done. Output root: %s elapsed_sec=%.1f", args.out_root, elapsed)


if __name__ == "__main__":
    main()
