#!/usr/bin/env bash
set -euo pipefail

cd /data_nvme/vocos_training

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
MFA_JOBS="${MFA_JOBS:-3}"
MAX_ITEMS="${MAX_ITEMS:-20000}"
DEVICE="${DEVICE:-cuda}"

CORPUS_DIR="mfa_corpus/vi_val10k"
ALIGN_DIR="mfa_aligned/vi_val10k"
TONE_MANIFEST="manifests/tone_eval_val.tsv"
EVAL_CKPT="checkpoints/tone_classifier/eval.pt"

mkdir -p run_logs mfa_aligned manifests eval_reports

ALIGN_LOG="run_logs/mfa_align_val10k_${STAMP}.log"
EXTRACT_LOG="run_logs/extract_tone_eval_val_${STAMP}.log"
EVAL_LOG="run_logs/evaluate_tone_val_${STAMP}.log"

count_textgrids() {
  find "$ALIGN_DIR" -maxdepth 1 -name '*.TextGrid' 2>/dev/null | wc -l
}

TEXTGRID_COUNT="$(count_textgrids)"
if [ "$TEXTGRID_COUNT" -lt 9000 ]; then
  source /home/huy/miniconda3/etc/profile.d/conda.sh
  conda activate mfa
  mfa align "$CORPUS_DIR" vietnamese_mfa vietnamese_mfa "$ALIGN_DIR" --overwrite -j "$MFA_JOBS" 2>&1 | tee "$ALIGN_LOG"
  conda deactivate
else
  echo "skip_mfa_align existing_textgrids=$TEXTGRID_COUNT" | tee "$ALIGN_LOG"
fi

source /home/huy/miniconda3/etc/profile.d/conda.sh
conda activate py310

python extract_tone_segments.py \
  --mfa-manifest "$CORPUS_DIR/mfa_manifest.tsv" \
  --textgrid-dir "$ALIGN_DIR" \
  --output "$TONE_MANIFEST" 2>&1 | tee "$EXTRACT_LOG"

python evaluate_tone_vocos_set.py \
  --tone-manifest "$TONE_MANIFEST" \
  --tone-evaluator-ckpt "$EVAL_CKPT" \
  --output-dir eval_reports \
  --output-prefix tone_val \
  --max-items "$MAX_ITEMS" \
  --device "$DEVICE" \
  --checkpoint baseline=checkpoints/vocos_mp3/best.pt \
  --checkpoint plus_c=checkpoints/ablations/plus_c/best.pt \
  --checkpoint plus_cb=checkpoints/ablations/plus_cb/best.pt \
  --checkpoint plus_cba=checkpoints/ablations/plus_cba/best.pt 2>&1 | tee "$EVAL_LOG"
