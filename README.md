# Tone-aware Vocos

Reference code, configurations, and analysis for **tone-aware neural vocoding for tonal
languages**, using Vietnamese as a case study. The paper trains a
[Vocos](https://github.com/gemelo-ai/vocos) generator with three *training-time-only*
objectives that make the vocoder reproduce lexical-tone F0 more faithfully, at **zero added
inference cost** (the auxiliary modules are discarded at deployment; the generator is byte-for-byte
a stock Vocos model).

> This repository accompanies the manuscript and is provided for reproducibility. The training
> corpus, model checkpoints, and audio are **not** redistributed here (licensing / size); this
> repo contains the code, configs, and the aggregate + per-item metric tables behind the paper's
> tables and figures.

## The three tone-aware components

The vocoder architecture (log-mel front end → ConvNeXt backbone → iSTFT head) is **unchanged**
from Vocos. Tone-awareness is added only in the training objective, as an ablation ladder:

| Config | C — F0-weighted MR-STFT | B — Constant-Q tone critic | A — tone-classifier feature loss |
|---|:---:|:---:|:---:|
| `baseline` | – | – | – |
| `plus_c`   | ✓ | – | – |
| `plus_cb`  | ✓ | ✓ | – |
| `plus_cba` | ✓ | ✓ | ✓ |

- **C — F0-weighted multi-resolution STFT loss** (`vocos_train/tone_losses.py`,
  `WeightedMultiResolutionSTFTLoss`): up-weights spectral error in the low band that carries
  tonal F0 (3× below 1 kHz, FFT sizes 512/1024/2048).
- **B — band-limited Constant-Q tone discriminator** (`vocos_train/tone_discriminator.py`,
  `CQTToneDiscriminator`): an adversarial critic over a log-frequency CQT restricted to the
  tone band (fmin 80 Hz, 48 bins @ 24 bins/octave ⇒ ≈80–320 Hz).
- **A — frozen tone-classifier feature loss** (`vocos_train/tone_losses.py`,
  `ToneFeatureMatchingLoss`): feature-matching against a pretrained tone classifier.
  **This is a negative ablation** — it does not help and slightly regresses F0 (see results);
  it is reported for completeness, not recommended.

## Headline results (20,000 held-out, unseen-voice syllable segments)

| System | Tone-classifier acc. | F0 RMSE (Hz) ↓ |
|---|:---:|:---:|
| `baseline`           | 57.48 % | 3.498 |
| `plus_c`             | 57.49 % | 3.340 |
| **`plus_cb`** (ours) | **57.73 %** | **3.271**  (−6.5 %) |
| `plus_cba`           | 57.37 % | 3.402 |

The workhorse is **C**; the **C+B** recipe gives the best F0 (paired Wilcoxon + Holm–Bonferroni
significant, and robust to voice-level cluster bootstrap). Tone-classifier **accuracy is
essentially unchanged** — the gain is in F0 fidelity, not in categorical tone identity. Adding
**A** regresses. Full statistics, confidence intervals, per-tone breakdowns, and external
baselines (Vocos-EN, BigVGAN-v2) are in [`results/`](results/).

## Repository layout

```
vocos_train/        core training library (the C/B/A components live here)
  tone_losses.py        C (F0-weighted MR-STFT) + A (tone-classifier feature loss)
  tone_discriminator.py B (Constant-Q tone critic)
  tone_classifier.py    BiGRU tone classifier (loss backbone + independent evaluator)
  model.py trainer.py data.py config.py vietnamese_tones.py
configs/ablations/  baseline / plus_c / plus_cb / plus_cba  (1 M steps each)
train.py                        training entry point
train_tone_classifier.py        train the tone classifier used by A and by evaluation
prepare_manifest.py prepare_mfa_corpus.py extract_tone_segments.py   data prep
evaluate_tone_vocos*.py aggregate_tone_eval_shards.py run_val_tone_eval.sh   evaluation
export_vocos_for_zipvoice.py synthesize.py    export / inference helpers
tools/              eval-pack and ZipVoice pairing helpers
eval/               analysis scripts (stats, figures, efficiency, checkpoint stability)
results/            aggregate tables + anonymized per-item metrics (paper tables/figures)
docs/               method notes and the validation runbook
```

## Reproducing

```bash
pip install -r requirements.txt          # torch, torchaudio, vocos==0.1.0, nnAudio, ...
# 1. prepare manifests + MFA corpus from your own 24 kHz corpus (see docs/ and prepare_*.py)
# 2. train the tone classifier used by A and by evaluation
python train_tone_classifier.py
# 3. train each rung of the ablation ladder (edit data.audio_root / manifests first)
python train.py --config configs/ablations/plus_cb.yaml
# 4. extract forced-aligned tone segments and score
bash run_val_tone_eval.sh
# 5. regenerate the paper's statistics and figures from results/
python eval/stats_analysis.py
python eval/make_figures.py
```

Exact package versions used for the reported numbers are pinned in
[`results/requirements.lock`](results/requirements.lock); GPU is in
[`results/gpu.txt`](results/gpu.txt) (single RTX 4090, 24 GB).

## Notes on the released data

- The per-item files (`results/segment_peritem_*.tsv.gz`, `results/utterance_metrics_merged.tsv`)
  are **voice-anonymized**: the 80 held-out speakers are labelled `v001`–`v080`, which preserves
  voice-level grouping for the cluster bootstrap while not disclosing the source voices.
- Audio, checkpoints, and the training corpus are not included.

## License

Code is released under the [MIT License](LICENSE). It builds on
[Vocos](https://github.com/gemelo-ai/vocos) (MIT), which must be installed as a dependency.

## Citation

If you use this code, please cite the accompanying paper (details to follow upon publication):

```bibtex
@article{toneawarevocos,
  title  = {Tone-aware neural vocoding for tonal languages: a Vietnamese case study},
  author = {Author, A. and others},
  year   = {2026},
  note   = {Under review}
}
```
