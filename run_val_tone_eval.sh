#!/usr/bin/env bash
set -euo pipefail

# Repo-relative, not the author's absolute path: this script is published, and
# `cd /data_nvme/vocos_training` aborted on line 4 for every other machine.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Overridable so a reproducer's conda does not have to live in the author's home.
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
MFA_JOBS="${MFA_JOBS:-3}"
MAX_ITEMS="${MAX_ITEMS:-20000}"
DEVICE="${DEVICE:-cuda}"

# The paper's baseline is the vocos_mp3 run (configs/vocos_mp3.yaml). Retraining it from
# configs/ablations/baseline.yaml writes elsewhere, so override this to point at your own:
#   BASELINE_CKPT=checkpoints/ablations/baseline/best.pt ./run_val_tone_eval.sh
BASELINE_CKPT="${BASELINE_CKPT:-checkpoints/vocos_mp3/best.pt}"

CORPUS_DIR="mfa_corpus/vi_val10k"
ALIGN_DIR="mfa_aligned/vi_val10k"

# The published numbers were scored on the OVERLAP-FILTERED manifest (205,282 segments), not
# the raw one this script's extract step writes (210,815). The filter that produces it is not
# part of this repository, so if it is absent we fall back and say so loudly rather than
# silently scoring a different set.
TONE_MANIFEST_PAPER="manifests/tone_eval_val_no_overlap.tsv"
TONE_MANIFEST_RAW="manifests/tone_eval_val.tsv"
SEED="${SEED:-20260706}"
EVAL_CKPT="checkpoints/tone_classifier/eval.pt"

mkdir -p run_logs mfa_aligned manifests eval_reports

ALIGN_LOG="run_logs/mfa_align_val10k_${STAMP}.log"
EXTRACT_LOG="run_logs/extract_tone_eval_val_${STAMP}.log"
EVAL_LOG="run_logs/evaluate_tone_val_${STAMP}.log"

count_textgrids() {
  # `|| true`: on a fresh checkout ALIGN_DIR does not exist, find exits 1, and pipefail
  # would propagate that through the command substitution and kill the script under `set -e`
  # before the "run MFA align" branch is ever reached.
  find "$ALIGN_DIR" -maxdepth 1 -name '*.TextGrid' 2>/dev/null | wc -l || true
}

TEXTGRID_COUNT="$(count_textgrids || echo 0)"
if [ "$TEXTGRID_COUNT" -lt 9000 ]; then
  source "$CONDA_SH"
  conda activate mfa
  mfa align "$CORPUS_DIR" vietnamese_mfa vietnamese_mfa "$ALIGN_DIR" --overwrite -j "$MFA_JOBS" 2>&1 | tee "$ALIGN_LOG"
  conda deactivate
else
  echo "skip_mfa_align existing_textgrids=$TEXTGRID_COUNT" | tee "$ALIGN_LOG"
fi

source "$CONDA_SH"
conda activate py310

python extract_tone_segments.py \
  --mfa-manifest "$CORPUS_DIR/mfa_manifest.tsv" \
  --textgrid-dir "$ALIGN_DIR" \
  --output "$TONE_MANIFEST_RAW" 2>&1 | tee "$EXTRACT_LOG"

if [ -f "$TONE_MANIFEST_PAPER" ]; then
  TONE_MANIFEST="$TONE_MANIFEST_PAPER"
else
  TONE_MANIFEST="$TONE_MANIFEST_RAW"
  echo "WARNING: $TONE_MANIFEST_PAPER not found. Scoring $TONE_MANIFEST_RAW instead." >&2
  echo "         The published numbers used the filtered manifest, so yours WILL differ." >&2
fi

python evaluate_tone_vocos_set.py \
  --tone-manifest "$TONE_MANIFEST" \
  --tone-evaluator-ckpt "$EVAL_CKPT" \
  --output-dir eval_reports \
  --output-prefix tone_val_no_overlap \
  --max-items "$MAX_ITEMS" \
  --shuffle \
  --seed "$SEED" \
  --device "$DEVICE" \
  --checkpoint baseline="$BASELINE_CKPT" \
  --checkpoint plus_c=checkpoints/ablations/plus_c/best.pt \
  --checkpoint plus_cb=checkpoints/ablations/plus_cb/best.pt \
  --checkpoint plus_cba=checkpoints/ablations/plus_cba/best.pt 2>&1 | tee "$EVAL_LOG"
