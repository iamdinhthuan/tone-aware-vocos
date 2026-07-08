# Tone-Aware Vocos — Autonomous Validation Runbook

> **Đối tượng đọc: agent Claude Code chạy tự động trên server.**
> Nhiệm vụ: validate & benchmark 5 phiên bản vocoder đã train xong, tạo ra bộ kết quả
> **publication-ready** cho tạp chí *Engineering Applications of Artificial Intelligence* (Elsevier, Q1).
> Chuẩn EAAI đòi hỏi: ablation đầy đủ, kiểm định thống kê **có hiệu chỉnh đa so sánh**,
> benchmark **hiệu năng/RTF**, và **reproducibility** (log môi trường, seed, config).
>
> Agent PHẢI tự chạy hết Phase 1→8, ghi log, và cuối cùng sinh `VALIDATION_REPORT.md`.

---

## 0. QUY TẮC AUTONOMY (đọc trước khi làm)

1. **Chạy tuần tự Phase 1 → 8.** Không nhảy phase. Mỗi phase có *success criteria* — chỉ sang phase
   sau khi đạt.
2. **Ghi log mọi thứ** vào `validation_out/logs/<phase>.log`. Mọi lệnh in ra cả stdout lẫn stderr.
3. **Khi thiếu input hoặc phase fail:** thử tự khắc phục theo mục "on-failure" của phase đó. Nếu vẫn
   fail → **DỪNG, ghi rõ vào `validation_out/BLOCKERS.md`** (thiếu gì, lệnh gì lỗi, log ở đâu) rồi
   báo người dùng. **Không bịa số liệu, không bỏ qua âm thầm.**
4. **Không sửa checkpoint/model đã train.** Chỉ đọc để inference.
5. **Xác định NaN rõ ràng:** nếu một metric ra NaN cho >20% số câu → coi là fail metric đó, ghi vào
   BLOCKERS, không đưa vào bảng như thể hợp lệ.
6. **Reproducibility:** đầu Phase 1 lưu `pip freeze`, `nvidia-smi`, git commit hash, nội dung mọi
   config vào `validation_out/repro/`.
7. Việc cần **con người** (nghe MOS) thì agent chỉ **chuẩn bị vật liệu**, KHÔNG tự chấm điểm.

---

## 1. INPUT BẮT BUỘC — AGENT VERIFY TRƯỚC (Phase 0/manifest)

Trước khi làm gì, kiểm tra tồn tại. Thiếu bất kỳ mục ⛔ nào → ghi BLOCKERS + dừng.

| Mục | Đường dẫn kỳ vọng | Bắt buộc |
|-----|-------------------|:---:|
| 5 checkpoint vocoder | `checkpoints/{baseline_vi,C,B,A,ABC}/*.ckpt` | ⛔ |
| Test set (speaker **unseen**) | `data/test_unseen_speakers/**/*.wav` | ⛔ |
| Tone classifier **eval** (khác classifier trong loss A) | `checkpoints/tone_classifier_eval.ckpt` | ⚠️ (thiếu thì bỏ metric tone, ghi rõ) |
| Nhãn thanh + alignment cho test set | `data/test_tone_labels/*.json` (mỗi câu: list [start,end,tone_id]) | ⚠️ (thiếu thì confusion matrix chạy chế độ pseudo-label, xem Phase 3) |
| Config mel dùng khi train | `configs/*.yaml` | ⛔ |
| Script benchmark có sẵn | `benchmark_vocoders.py` (đã cung cấp) | ⛔ |

**Kiểm tra nhất quán quan trọng:** đọc mel config từ `configs/` và bảo đảm **cả 5 model dùng chung
một mel config**. Nếu lệch → dừng, báo BLOCKERS (so sánh sẽ vô hiệu).

Ghi kết quả verify vào `validation_out/manifest_check.md`.

---

## 2. MÔI TRƯỜNG

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U numpy scipy pandas soundfile librosa matplotlib seaborn pyyaml tqdm
pip install torch torchaudio            # bản khớp CUDA của server (5090 -> cu12x)
pip install pesq pystoi pyworld pysptk pyloudnorm nnAudio
pip install speechmos                   # UTMOS; nếu fail, harness tự fallback torch.hub
```
Ghi `pip freeze > validation_out/repro/requirements.lock`.
Kiểm tra GPU: `python -c "import torch;print(torch.cuda.get_device_name(0))"` → phải thấy 5090.

**On-failure:** nếu `nnAudio`/`pysptk` không cài được → vẫn tiếp tục nhưng metric phụ thuộc (không có).
Ghi rõ metric nào bị thiếu.

---

## 3. CONFIG (agent tạo `validation_config.yaml`)

```yaml
sr: 24000
mel: {n_fft: 1024, hop_length: 256, win_length: 1024, n_mels: 100, f_min: 0.0, f_max: null, power: 1.0}
models:
  baseline-vi: checkpoints/baseline_vi/last.ckpt
  C:           checkpoints/C/last.ckpt
  B:           checkpoints/B/last.ckpt
  A:           checkpoints/A/last.ckpt
  A+B+C:       checkpoints/ABC/last.ckpt
baseline_name: baseline-vi
external_baselines:                       # Phase 1B — tải & chạy cùng test set
  Vocos-pretrained: {type: hf, id: "charactr/vocos-mel-24khz"}          # BẮT BUỘC
  BigVGAN-v2:       {type: hf, id: "nvidia/bigvgan_v2_24khz_100band_256x"}
  HiFi-GAN-univ:    {type: hf, id: "<universal hifigan ckpt nếu có>"}   # bỏ nếu không tìm được
test_wavs_glob: data/test_unseen_speakers/**/*.wav
tone_clf_eval_ckpt: checkpoints/tone_classifier_eval.ckpt
tone_labels_dir: data/test_tone_labels
target_lufs: -27.0
out_dir: validation_out
n_test_min: 200          # nếu ít hơn -> cảnh báo power thống kê yếu
tone_ids: {0: ngang, 1: sac, 2: huyen, 3: hoi, 4: nga, 5: nang}
```
Agent sửa `benchmark_vocoders.py` để đọc từ file config này (thay vì hằng số cứng), và **điền 2 hàm
phụ thuộc fork** `load_vocoder()` + `synthesize()` bằng cách đọc code train trong repo để biết API.

**On-failure (không suy ra được API vocoder):** thử `VocosExp.load_from_checkpoint`; nếu không có,
grep trong repo tìm class inference (`pretrained.py`/`Vocos`). Vẫn bí → BLOCKERS.

---

## PHASE 1 — Sanity checks
**Mục tiêu:** đảm bảo mỗi model load & synth được, output không hỏng.

- Load lần lượt 5 model. Với 3 câu test bất kỳ: synth → kiểm tra output không NaN/inf, RMS > 1e-4,
  độ dài hợp lý (±1 frame so với input).
- Lưu 3 cặp (real, synth) mỗi model vào `validation_out/sanity/<model>/` để nghe kiểm.

**Success:** cả 5 model qua hết. **On-failure:** model nào hỏng → BLOCKERS (ghi tên + traceback), dừng.

---

## PHASE 1B — External pretrained / SOTA baselines (BẮT BUỘC cho EAAI)
**Mục tiêu:** reviewer Q1 sẽ hỏi "so với vocoder pretrained/SOTA công khai thì sao?". Phải đưa các
baseline ngoài vào **cùng một bảng, cùng test set, cùng mel config resynthesis**. Đây là bằng chứng
(a) train lại trên 7000h tiếng Việt đáng giá, (b) method thắng SOTA hiện có.

**Hai nhóm baseline cần thêm (đừng gộp lẫn ý nghĩa):**

1. **Vocos pretrained gốc (English, off-the-shelf)** — chứng minh giá trị của việc train tiếng Việt.
   - Tải: `from vocos import Vocos; Vocos.from_pretrained("charactr/vocos-mel-24khz")`.
2. **SOTA universal vocoder có checkpoint công khai** — định vị so với SOTA:
   - **BigVGAN-v2**: `nvidia/bigvgan_v2_24khz_100band_256x` (HuggingFace, khớp 24kHz/100-mel — tiện).
   - **HiFi-GAN (universal)**: checkpoint universal công khai (vd `universal_v1`).
   - (Tùy chọn) một vocoder tiếng Việt/đa ngữ khác nếu tìm được checkpoint.

**Điểm mel config — QUAN TRỌNG:** các baseline ngoài **thường yêu cầu mel config RIÊNG** (n_fft,
n_mels, fmin/fmax, chuẩn hóa) khác của bạn. Resynthesis phải trích mel **theo đúng config mỗi model
yêu cầu**, KHÔNG ép chúng dùng mel của bạn (sẽ bất công ngược, làm baseline tệ giả tạo). Nghĩa là:
- 5 model của bạn: dùng mel config của bạn.
- Mỗi baseline ngoài: dùng mel config gốc của nó (đọc từ card/model repo).
- Điểm so sánh chung & công bằng là **cùng audio thật đầu vào và cùng audio thật để so ở đầu ra** —
  metric (PESQ/F0/…) tính trên waveform, không phụ thuộc mel, nên vẫn so được. Ghi rõ trong report
  rằng mỗi hệ dùng mel front-end gốc của nó (đây là cách chuẩn khi so cross-vocoder).

**Việc agent làm:**
- Viết `external_baselines.py`: mỗi baseline một wrapper `synthesize_external(name, audio)` trả về
  waveform 24kHz (resample nếu SR khác), dùng đúng feature extractor gốc của baseline đó.
- Chạy resynthesis trên **cùng test set** → nối kết quả vào `per_utterance.csv` như các model khác
  (thêm cột model = "Vocos-pretrained", "BigVGAN-v2", "HiFi-GAN-univ").
- Các baseline này cũng đi qua Phase 3 (tone), Phase 4 (efficiency), Phase 5 (stats), Phase 6 (figures).

**On-failure (không tải được checkpoint do mạng/thiếu lib):** ghi vào BLOCKERS baseline nào tải hụt,
tiếp tục với các baseline tải được. **Tối thiểu phải có Vocos-pretrained gốc** (cùng họ kiến trúc,
so sánh sát nhất); nếu ngay cả cái này cũng không tải được → dừng hỏi người dùng.

**Diễn giải kỳ vọng (agent ghi vào report):**
- `baseline-vi` **nên thắng** Vocos-pretrained gốc (khớp trải nghiệm "sạch hơn, ít rè hơn") → xác nhận
  giá trị train tiếng Việt.
- `A+B+C` **nên thắng** cả BigVGAN-v2/HiFi-GAN ở **metric thanh điệu** (F0/Tone), dù có thể sát nhau ở
  metric chất lượng chung → đó chính là đóng góp của paper.
- Nếu một baseline ngoài thắng bạn ở metric chung (vd BigVGAN-v2 UTMOS cao hơn): **báo cáo trung thực**,
  và làm nổi bật rằng ưu thế của bạn nằm ở thanh điệu + hiệu năng (RTF) — đừng giấu.

**Success:** ≥2 baseline ngoài (bắt buộc có Vocos-pretrained) đã nằm trong `per_utterance.csv`.

---

## PHASE 2 — Objective resynthesis benchmark
**Mục tiêu:** bảng metric chính + per-utterance CSV cho significance.

- Chạy `python benchmark_vocoders.py` (đã đọc `validation_config.yaml`).
- Sinh `validation_out/per_utterance.csv` và `validation_out/summary.csv`.
- Metric tối thiểu phải có: **UTMOS, PESQ, ESTOI, MCD_dB, F0_RMSE_cents, F0_corr, VUV_err, Tone_agree**.

**Success:** mỗi metric có ≥80% câu giá trị hợp lệ (không NaN) cho cả 5 model.
**On-failure metric X:** ghi X vào BLOCKERS, tiếp tục các metric khác (không dừng cả phase vì 1 metric).

**Kiểm tra tỉnh táo (agent tự đánh giá, ghi nhận xét):**
- Baseline-vi nên **kém hơn** ở F0/Tone so với các bản tone-aware nếu method có tác dụng. Nếu ngược
  lại (baseline tốt hơn mọi mặt) → không phải lỗi, nhưng ghi chú nổi bật để người dùng biết.

---

## PHASE 3 — Per-tone breakdown + Confusion matrix
**Mục tiêu:** bằng chứng cốt lõi của câu chuyện "cải thiện thanh điệu". Agent **tự viết**
`analyze_tone.py` theo spec sau.

**Chế độ A — có nhãn thanh + alignment (`tone_labels_dir` tồn tại):** (ưu tiên)
- Với mỗi câu & mỗi model: chạy tone classifier eval trên **synth**, gán nhãn dự đoán theo từng
  segment âm tiết (dùng biên [start,end] trong label json). So với **ground-truth tone_id**.
- Xuất:
  - `tone_confusion_<model>.csv` — ma trận 6×6 (hàng = thật, cột = dự đoán).
  - `per_tone_accuracy.csv` — accuracy theo từng thanh × từng model.
  - `per_tone_f0_rmse.csv` — F0 RMSE (cents) theo từng thanh × model (chỉ voiced). **Nga/nang riêng**
    vì creaky — đây là chỗ dễ khoe cải thiện nhất.

**Chế độ B — không có nhãn (pseudo-label):**
- Dùng dự đoán tone classifier trên **audio thật** làm pseudo-ground-truth, so với dự đoán trên synth.
- Đánh dấu rõ trong output là "pseudo-label (no human tone annotation)". Yếu hơn nhưng vẫn trình bày được.

**Success:** sinh đủ file confusion + per-tone cho cả 5 model.
**On-failure (không có tone classifier eval):** bỏ Phase 3, ghi BLOCKERS "cần tone_classifier_eval.ckpt".

---

## PHASE 4 — Efficiency benchmark (quan trọng cho EAAI)
**Mục tiêu:** EAAI là tạp chí *ứng dụng* → cần số liệu thực dụng. Agent viết `bench_efficiency.py`.

- **Params (generator only, dùng khi inference):** đếm số tham số + kích thước MB. (Không tính
  discriminator/tone classifier vì chúng không dùng lúc synth.)
- **RTF (Real-Time Factor)** trên **5090**: synth 50 câu, đo `tổng_thời_gian_synth / tổng_thời_lượng_audio`.
  Warm-up 5 lần trước khi đo. Báo trung bình ± std. RTF < 1 = nhanh hơn thời gian thực.
- **RTF trên CPU** (1 luồng) nếu khả thi — cho thấy tính triển khai được.
- **Peak VRAM** khi synth (torch.cuda.max_memory_allocated).
- Xuất `efficiency.csv`: model × {params_M, size_MB, RTF_gpu, RTF_cpu, vram_MB}.

**Lưu ý:** nếu A/B/C **không đổi generator** (chỉ thêm loss/discriminator lúc train) thì params & RTF
của baseline và các bản gần **bằng nhau** — điều này TỐT (cải thiện chất lượng **miễn phí** lúc inference).
Agent nêu rõ điểm này trong report; đó là một selling point mạnh cho EAAI.
Nếu bản D (F0 conditioning) có mặt → params sẽ khác, báo cáo trung thực.

**Success:** `efficiency.csv` đầy đủ 5 model.

---

## PHASE 5 — Phân tích thống kê (có hiệu chỉnh đa so sánh)
**Mục tiêu:** significance đạt chuẩn tạp chí. Agent viết `stats_analysis.py`.

- Với mỗi metric: **paired Wilcoxon signed-rank** giữa `baseline-vi` và từng bản (C, B, A, A+B+C)
  trên các câu chung. **Thêm** so sánh `A+B+C` vs từng **external baseline** (Vocos-pretrained,
  BigVGAN-v2, HiFi-GAN-univ) — đây là bằng chứng "thắng SOTA" cho reviewer.
- **Bắt buộc hiệu chỉnh đa so sánh:** vì so 4 bản × nhiều metric → dùng **Holm–Bonferroni** trên tập
  p-value của mỗi metric. Báo cả p thô và p hiệu chỉnh.
- Báo **effect size** (matched-pairs rank-biserial correlation hoặc Cliff's delta) — reviewer EAAI
  hay hỏi "khác biệt có *đáng kể về thực chất* không", không chỉ p<0.05.
- Báo **mean ± 95% CI** (bootstrap 1000 lần cho khoảng tin cậy trung bình hiệu).
- Xuất `stats_summary.csv` + `stats_summary.md` (bảng đọc được).

**Success:** mọi cặp (metric × model) có p_raw, p_holm, effect_size, CI.
**Cảnh báo power:** nếu số câu < `n_test_min` (200) → ghi cảnh báo "low statistical power" trong report.

---

## PHASE 6 — Hình vẽ publication-ready
**Mục tiêu:** hình cho paper (300 DPI, vector PDF + PNG). Agent viết `make_figures.py`, lưu vào
`validation_out/figures/`.

1. **F0 contour overlay:** chọn 3–4 câu chứa thanh nga/nang; vẽ F0 (thật vs baseline vs A+B+C) trên
   cùng trục thời gian → cho thấy bản tone-aware bám contour tốt hơn ở đoạn creaky.
2. **Spectrogram cận cảnh đoạn creaky:** thật / baseline / A+B+C cạnh nhau → cho thấy giảm artifact.
3. **Per-tone F0 RMSE bar chart:** 6 thanh × 5 model (từ Phase 3).
4. **Confusion matrix heatmap:** baseline vs A+B+C cạnh nhau (từ Phase 3).
5. **Bar chart tổng hợp:** metric chính (F0_RMSE_cents, Tone_agree, UTMOS) qua **cả external baselines
   + 5 model của bạn**, error bar 95% CI, tô màu khác nhóm (external vs ours) để dễ đọc.
6. **RTF vs quality scatter:** trục X = RTF, trục Y = chất lượng (vd Tone_agree hoặc UTMOS), chấm cả
   external baselines + ours → cho thấy bạn ở góc "nhanh + tốt", cải thiện gần như miễn phí.

Mỗi hình lưu cả `.pdf` (vector, cho LaTeX) và `.png`. Font ≥ 9pt, colorblind-safe palette.

**Success:** đủ 6 nhóm hình.

---

## PHASE 7 — Chuẩn bị vật liệu nghe chủ quan (con người chấm)
**Mục tiêu:** EAAI thường cần MOS/preference. Agent KHÔNG chấm — chỉ chuẩn bị. Viết `prep_listening_test.py`.

- Chọn **ngẫu nhiên** N=25 câu test (ưu tiên câu chứa nga/nang), mỗi câu có: reference + 5 bản synth.
- **Chuẩn hóa loudness** tất cả về -27 LUFS, **randomize thứ tự & ẩn tên model** (mã hóa file), lưu
  bảng ánh xạ bí mật `listening_test/key.csv`.
- Tạo `listening_test/` gồm audio ẩn danh + một **HTML MUSHRA/MOS đơn giản** (thang 1–5, có nút play,
  ô nhập điểm) + hướng dẫn cho người nghe bản ngữ, tập trung câu hỏi **"nghe có đúng thanh không"**.
- Ghi rõ trong report: "cần ≥15 người nghe bản ngữ; kết quả điền sau."

**Success:** thư mục `listening_test/` sẵn sàng phát cho người nghe.

---

## PHASE 8 — Tự lắp ráp báo cáo cuối
**Mục tiêu:** `validation_out/VALIDATION_REPORT.md` — tổng hợp mọi thứ, sẵn để trích vào paper.

Bố cục report agent phải sinh:
1. **Tóm tắt & verdict:** bản nào tốt nhất theo từng nhóm metric; cải thiện có significant (p_holm) không.
2. **Bảng chính (gộp external + ablation nội bộ):** chia 2 nhóm hàng có kẻ ngăn —
   *(a) External baselines*: Vocos-pretrained, BigVGAN-v2, HiFi-GAN-univ;
   *(b) Ours (ablation)*: baseline-vi, +C, +B, +A, A+B+C.
   Cột: {UTMOS, PESQ, ESTOI, MCD, F0_RMSE_cents, F0_corr, VUV_err, Tone_agree}, mean±CI,
   **in đậm giá trị tốt nhất mỗi cột toàn bảng**, đánh dấu significance (p_holm<0.05) so với baseline-vi.
   Kèm 1–2 câu diễn giải: baseline-vi vs Vocos-pretrained (giá trị train tiếng Việt) và
   A+B+C vs SOTA ngoài (đóng góp thanh điệu + hiệu năng).
3. **Bảng per-tone** (accuracy + F0 RMSE theo 6 thanh) — nêu bật nga/nang.
4. **Bảng hiệu năng** (params, RTF gpu/cpu, VRAM) + nhận định "cải thiện gần như miễn phí lúc inference".
5. **Thống kê:** p_raw/p_holm/effect_size/CI cho mọi so sánh.
6. **Danh mục hình** (đường dẫn + caption gợi ý).
7. **Reproducibility appendix:** GPU, commit, seed, mel config, requirements.lock.
8. **Hạn chế & việc còn lại:** MOS chờ người chấm; cảnh báo power nếu N nhỏ; metric nào bị thiếu.
9. **BLOCKERS** (nếu có) đính kèm.

Đồng thời copy toàn bộ CSV/figure vào `validation_out/` gọn gàng và in ra cây thư mục cuối cùng.

---

## OUTPUT ARTIFACTS (agent phải tạo đủ)

```
validation_out/
├── VALIDATION_REPORT.md          <- deliverable chính
├── manifest_check.md
├── BLOCKERS.md                   <- rỗng nếu mọi thứ ổn
├── per_utterance.csv             <- gồm CẢ external baselines lẫn 5 model của bạn
├── summary.csv
├── stats_summary.{csv,md}
├── efficiency.csv
├── tone_confusion_<model>.csv
├── per_tone_accuracy.csv
├── per_tone_f0_rmse.csv
├── figures/            (*.pdf + *.png)
├── listening_test/     (audio ẩn danh + HTML + key.csv)
├── sanity/             (mẫu nghe nhanh)
├── logs/
└── repro/              (requirements.lock, gpu.txt, commit.txt, configs sao lưu)
```

---

## TIÊU CHÍ HOÀN THÀNH (definition of done)
Agent coi là XONG khi: Phase 1–8 chạy hết, `VALIDATION_REPORT.md` tồn tại và có đủ 9 mục,
mọi bảng có significance đã hiệu chỉnh, hình đã sinh, `listening_test/` sẵn sàng, và `BLOCKERS.md`
hoặc rỗng hoặc liệt kê rõ mọi vướng mắc còn lại. Sau đó in tóm tắt verdict ra stdout cho người dùng.

## KHI NÀO DỪNG HỎI NGƯỜI DÙNG (thay vì tự quyết)
- Thiếu checkpoint/test set/mel config lệch giữa các model (⛔ input).
- Không suy được API `load_vocoder/synthesize` từ repo.
- Baseline tốt hơn mọi bản tone-aware ở *tất cả* metric (bất thường — cần người xác nhận trước khi
  viết kết luận).
- Không tải được **Vocos-pretrained gốc** (baseline external tối thiểu bắt buộc).
Ngoài các trường hợp trên, agent tự xử lý và ghi chú, không hỏi vặt.
