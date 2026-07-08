# Tone-Aware Vocos — Implementation Spec (cho coding agent)

> Mục tiêu: cải tiến vocoder **Vocos** để tái tạo **thanh điệu tiếng Việt** trung thực hơn,
> tạo đóng góp method-level cho paper. Backbone Vocos giữ nguyên; thêm 3 thành phần
> tone-aware + 1 thành phần tùy chọn. **Không** đổi pipeline dữ liệu của Vocos (vẫn chỉ cần audio).
>
> Repo gốc tham chiếu: `gemelo-ai/vocos` (PyTorch Lightning). Mọi đường dẫn file bên dưới
> theo cấu trúc repo đó. Nếu fork khác → map tương ứng.

---

## 0. TL;DR những gì cần làm

| # | Thành phần | Loại | File mới / sửa | Độ ưu tiên |
|---|-----------|------|----------------|-----------|
| A | **Tone-identity perceptual loss** (classifier 6 thanh, freeze, dùng feature-matching) | Loss | `vocos/tone_classifier.py` (mới), `vocos/loss.py` (sửa), `vocos/experiment.py` (sửa) | P0 — đóng góp chính |
| B | **CQT tone-focused discriminator** (tập trung dải F0 80–400 Hz) | Discriminator | `vocos/discriminators.py` (sửa), `vocos/experiment.py` (sửa) | P0 — đóng góp chính |
| C | **F0-weighted multi-resolution STFT loss** (re-weight dải tần thấp) | Loss | `vocos/loss.py` (sửa) | P1 — rẻ, nên có |
| D | **F0 + tone-embedding conditioning** vào backbone | Kiến trúc | `vocos/models.py`, `feature_extractors.py`, `dataset.py` | P2 — stretch goal |

**Nguyên tắc**: A, B, C **không cần** thay đổi dataset (vocoder vẫn train unsupervised trên audio).
Chỉ thành phần A cần một classifier được **pretrain riêng** trên dữ liệu có nhãn thanh điệu,
sau đó **freeze** khi train vocoder. D mới cần đưa F0/nhãn thanh vào dataloader.

---

## 1. Bản đồ kiến trúc Vocos gốc (đọc trước khi sửa)

```
vocos/
├── models.py            # VocosBackbone (ConvNeXt blocks) — generator core
├── heads.py             # ISTFTHead: dự đoán magnitude+phase -> iSTFT
├── feature_extractors.py# MelSpectrogramFeatures: audio -> mel (input của generator)
├── discriminators.py    # MultiPeriodDiscriminator (MPD), MultiResolutionDiscriminator (MRD)
├── loss.py              # MelSpecReconstructionLoss, GeneratorLoss, DiscriminatorLoss,
│                        #   FeatureMatchingLoss
├── spectral_ops.py      # ISTFT
├── modules.py           # ConvNeXtBlock, AdaLayerNorm...
├── experiment.py        # VocosExp (LightningModule) — VÒNG LẶP TRAIN nằm ở đây
├── dataset.py           # VocosDataset (chỉ load & cắt audio waveform)
└── pretrained.py        # class Vocos (inference)
train.py
configs/*.yaml
```

**Luồng generator**: `audio -> MelSpectrogramFeatures -> VocosBackbone -> ISTFTHead -> audio_hat`.

**Vòng lặp train** (trong `experiment.py`, dùng manual optimization, 2 optimizer):
- Bước discriminator: tính `DiscriminatorLoss` cho MPD + MRD trên (real, fake).
- Bước generator: tổng của
  - `mel_loss * mel_loss_coeff`
  - adversarial gen loss (MPD + MRD)
  - `feature_matching_loss` (MPD + MRD)

> **Hook chính cho mọi thay đổi loss/discriminator nằm trong `experiment.py`**, ở 2 nhánh
> discriminator-step và generator-step. Agent phải tìm 2 nhánh này (thường là `training_step`
> với kiểm tra optimizer index, hoặc 2 hàm con). Ghi log từng loss term riêng để debug cân bằng.

---

## 2. Thành phần A — Tone-Identity Perceptual Loss (P0)

### Ý tưởng
Train sẵn một **tone classifier** trên audio tiếng Việt thật (phân biệt 6 thanh). **Freeze** nó.
Khi train vocoder, đưa cả `audio_real` và `audio_hat` qua classifier, lấy **feature map các lớp
trung gian**, rồi tính **feature-matching loss** giữa chúng (giống perceptual loss kiểu VGG, hoặc
feature-matching trong GAN). Điều này ép vocoder giữ đúng các đặc trưng phân biệt thanh điệu —
kể cả khi mel loss đã "đủ tốt" về mặt phổ trung bình.

> Lý do dùng **feature-matching thay vì cross-entropy**: không cần nhãn/alignment cho dữ liệu
> train vocoder. Classifier chỉ cung cấp một "không gian đặc trưng nhạy thanh điệu". Nhãn chỉ
> cần khi *pretrain* classifier.

### A.1. File mới: `vocos/tone_classifier.py`

Kiến trúc gợi ý (nhẹ, đủ để bắt contour F0 + cấu trúc harmonic):

```python
import torch
import torch.nn as nn
import torchaudio

class ToneClassifier(nn.Module):
    """
    Input: waveform [B, T] (mono, sample_rate khớp vocoder, vd 24000).
    Output: logits [B, n_tones] (utterance-level) + intermediate feats (cho perceptual loss).
    n_tones = 6 (ngang, sắc, huyền, hỏi, ngã, nặng).
    """
    def __init__(self, sample_rate=24000, n_mels=80, n_tones=6, hidden=128):
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=1024, hop_length=256, n_mels=n_mels,
            f_min=40, f_max=sample_rate // 2, power=1.0,
        )
        # Conv front-end (trả về feature maps trung gian)
        self.conv = nn.ModuleList([
            nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU()),
            nn.Sequential(nn.Conv2d(32, 64, 3, stride=(2,1), padding=1), nn.BatchNorm2d(64), nn.GELU()),
            nn.Sequential(nn.Conv2d(64, 128, 3, stride=(2,1), padding=1), nn.BatchNorm2d(128), nn.GELU()),
        ])
        self.gru = nn.GRU(128 * (n_mels // 4), hidden, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * hidden, n_tones)

    def forward(self, wav, return_features=False):
        x = torch.log(self.melspec(wav).clamp_min(1e-5)).unsqueeze(1)  # [B,1,Mel,Time]
        feats = []
        for layer in self.conv:
            x = layer(x)
            feats.append(x)
        b, c, m, t = x.shape
        seq = x.permute(0, 3, 1, 2).reshape(b, t, c * m)  # [B,T, C*M]
        out, _ = self.gru(seq)
        logits = self.head(out.mean(dim=1))               # utterance-level
        if return_features:
            return logits, feats
        return logits
```

> Lưu ý: **MelSpectrogram trong classifier phải khả vi** và **không có `.detach()`** trên đường
> đi của `audio_hat` để gradient chảy về vocoder. Dùng `power=1.0` (magnitude) ổn định hơn.

### A.2. Pretrain classifier (script riêng, ngoài vòng train vocoder)

- **Dữ liệu**: corpus tiếng Việt có nhãn thanh điệu ở mức âm tiết (VIVOS, VLSP-TTS, InfoRE,
  vais1000... + forced alignment bằng **Montreal Forced Aligner** để lấy biên âm tiết → gán nhãn
  thanh theo dấu thanh của âm tiết).
- **Cách đơn giản hóa**: cắt audio theo từng âm tiết (dùng alignment) → mỗi mẫu 1 nhãn thanh
  trong {0..5}. Train cross-entropy. Đạt acc >85% trên tập test là dùng được.
- **Đầu ra**: checkpoint `tone_classifier.ckpt`. **Lưu lại MelSpectrogram config** để khớp.
- ⚠️ Thanh **ngã/nặng** có creaky voice → F0 nhiễu; augment + class weighting để tránh bias.

### A.3. Loss: thêm vào `vocos/loss.py`

```python
class ToneFeatureMatchingLoss(nn.Module):
    def __init__(self, tone_classifier):
        super().__init__()
        self.clf = tone_classifier
        for p in self.clf.parameters():
            p.requires_grad = False
        self.clf.eval()

    def forward(self, y, y_hat):  # y, y_hat: [B, T]
        with torch.no_grad():
            _, feats_real = self.clf(y, return_features=True)
        _, feats_fake = self.clf(y_hat, return_features=True)   # KHÔNG no_grad
        loss = 0.0
        for fr, ff in zip(feats_real, feats_fake):
            loss = loss + torch.mean(torch.abs(fr - ff))
        return loss / len(feats_real)
```

### A.4. Tích hợp vào `vocos/experiment.py`
- Trong `__init__`: load classifier từ checkpoint, tạo `self.tone_fm_loss = ToneFeatureMatchingLoss(clf)`.
  Đảm bảo classifier ở `eval()` và `requires_grad=False`, **không** đưa params của nó vào optimizer.
- Trong **generator-step**: cộng `self.tone_loss_coeff * self.tone_fm_loss(audio_real, audio_hat)`
  vào tổng generator loss. Log riêng `gen/tone_fm_loss`.
- Hệ số khởi điểm: `tone_loss_coeff = 1.0`. Tune trong khoảng `[0.5, 5.0]` sao cho cùng bậc độ
  lớn với feature-matching loss của MPD/MRD (xem log).

---

## 3. Thành phần B — CQT Tone-Focused Discriminator (P0)

### Ý tưởng
Thêm một discriminator chạy trên **Constant-Q Transform (CQT)** với trục tần log — rất hợp pitch —
và **giới hạn dải tần vào vùng F0 tiếng Việt (≈80–400 Hz, mở rộng tới ~1 kHz để lấy harmonic bậc thấp)**.
Discriminator này "nghe" được sai thanh mà MPD/MRD (vốn không ưu tiên dải F0) bỏ sót.

### B.1. Dependency
- Dùng **`nnAudio`** (`from nnAudio.features import CQT1992v2`) — CQT **khả vi**, chạy trên GPU.
  Thêm vào `requirements.txt`: `nnAudio`.

### B.2. File: thêm class vào `vocos/discriminators.py`

```python
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from nnAudio.features import CQT1992v2

class CQTToneDiscriminator(nn.Module):
    """
    CQT discriminator tập trung dải F0. Trả về (score, feature_maps) như MPD/MRD
    để dùng được feature_matching_loss có sẵn.
    """
    def __init__(self, sample_rate=24000, fmin=80.0, n_bins=48, bins_per_octave=24, hop=256):
        super().__init__()
        # fmin=80, 48 bins, 24 bins/octave -> phủ ~80 Hz đến ~80*2^(48/24)=~320 Hz (vùng F0 cốt lõi)
        self.cqt = CQT1992v2(sr=sample_rate, fmin=fmin, n_bins=n_bins,
                             bins_per_octave=bins_per_octave, hop_length=hop, output_format="Magnitude")
        ch = [1, 32, 64, 128, 128]
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(ch[i], ch[i+1], (3, 3), (1, 1), padding=(1, 1)))
            for i in range(len(ch) - 1)
        ])
        self.act = nn.LeakyReLU(0.1)
        self.post = weight_norm(nn.Conv2d(128, 1, (3, 1), (1, 1), padding=(1, 0)))

    def forward(self, x):                 # x: [B, T]
        z = torch.log(self.cqt(x).clamp_min(1e-5)).unsqueeze(1)  # [B,1,Bins,Frames]
        fmap = []
        for c in self.convs:
            z = self.act(c(z))
            fmap.append(z)
        score = self.post(z)
        return score, fmap
```

> Có thể làm **multi-scale**: 2–3 bản với `(fmin, bins_per_octave)` khác nhau (vd thêm bản
> bins_per_octave=36 phân giải mịn hơn). Gói trong một `nn.ModuleList` trả về list (scores, fmaps),
> theo đúng convention MRD.

### B.3. Tích hợp `experiment.py`
- `__init__`: `self.cqt_disc = CQTToneDiscriminator(...)`. **Đưa params của nó vào optimizer
  discriminator** (chung với MPD/MRD).
- **Discriminator-step**: tính `DiscriminatorLoss` cho CQT disc trên (real, fake), cộng vào disc loss.
- **Generator-step**: cộng adversarial gen loss + feature_matching_loss của CQT disc (tái dùng
  `GeneratorLoss`/`FeatureMatchingLoss` có sẵn). Log riêng `disc/cqt`, `gen/cqt_adv`, `gen/cqt_fm`.
- Hệ số: bắt đầu bằng **cùng trọng số** như MRD để cân bằng.

---

## 4. Thành phần C — F0-Weighted Multi-Resolution STFT Loss (P1)

### Ý tưởng
Mel loss + MR-STFT loss đối xử mọi bin tần như nhau. Re-weight để **phạt nặng hơn ở dải F0 thấp**.

### C.1. Sửa `vocos/loss.py` (hoặc thêm class mới)
- Với mỗi resolution STFT (n_fft trong {512, 1024, 2048}), tạo vector trọng số theo bin tần:
  cao ở dải 0–1 kHz, giảm dần lên cao. Ví dụ:

```python
def freq_weight(n_fft, sr, low_hz=1000.0, w_low=3.0, w_high=1.0):
    freqs = torch.linspace(0, sr / 2, n_fft // 2 + 1)
    w = torch.where(freqs <= low_hz, torch.tensor(w_low), torch.tensor(w_high))
    return w  # [F]  -> nhân vào |STFT(y)-STFT(y_hat)| trước khi mean, broadcast theo time
```

- Áp dụng vào spectral convergence + log-magnitude L1 của MR-STFT loss. Đăng ký buffer để khỏi
  tính lại. Bắt đầu `w_low=3.0`; ablate {1,2,3,5}.

> Đây là thay đổi **rẻ nhất** và thường cho cải thiện F0 RMSE rõ. Nên có để làm ablation đẹp.

---

## 5. Thành phần D — F0 + Tone-Embedding Conditioning (P2, stretch)

> Chỉ làm nếu A/B/C đã chạy ổn và còn thời gian. Đây là phần **đụng vào dataloader**.

### D.1. Dataset (`vocos/dataset.py`)
- Khi load mỗi đoạn audio, trích **F0 contour** đồng bộ khung với mel (hop_length giống mel).
  Dùng `torchcrepe` (khả vi, nhưng ở đây chỉ cần làm feature → có thể precompute) hoặc
  `pyworld` (Harvest/DIO) precompute & cache ra `.npy`.
- (Tùy chọn) nhãn thanh theo khung nếu có alignment; nếu không, bỏ tone-embedding, chỉ dùng F0.
- Trả về thêm `f0 [T_frames]` (log-F0 chuẩn hóa) + mask voiced/unvoiced.

### D.2. Generator (`vocos/feature_extractors.py` + `models.py`)
- Nối F0 (đã chiếu lên embedding nhỏ) vào input mel của `VocosBackbone`:
  `x = concat([mel, f0_proj], dim=channel)` → tăng `input_channels` của backbone tương ứng.
- (Tùy chọn) **harmonic source excitation**: sinh tín hiệu sine theo F0 (kiểu NSF) làm điều kiện
  phụ — giúp dải creaky (ngã/nặng) ổn định hơn. Đây là phần dễ thành novelty phụ nhưng tốn công.

> ⚠️ Related work cần phân biệt rõ trong paper: NSF, SiFi-GAN, harmonic-plus-noise vocoders.
> Định vị đóng góp ở "tonal-language fidelity" + kết hợp với A/B, **không** phải "pitch controllability".

---

## 6. Config (`configs/*.yaml`)

Thêm các khóa mới (giữ tương thích ngược, default tắt D):

```yaml
tone_aware:
  tone_classifier_ckpt: "checkpoints/tone_classifier.ckpt"
  tone_loss_coeff: 1.0            # A
  use_cqt_disc: true             # B
  cqt_fmin: 80.0
  cqt_n_bins: 48
  cqt_bins_per_octave: 24
  freq_weighted_stft: true       # C
  stft_low_hz: 1000.0
  stft_w_low: 3.0
  use_f0_conditioning: false     # D (stretch)
```

---

## 7. Đánh giá (BẮT BUỘC cho paper)

Implement script `evaluate.py` tính:
- **Chung**: UTMOS, PESQ, mel-cepstral distortion (MCD), ViSQOL nếu có.
- **Thanh điệu (điểm nhấn)**:
  - **F0 RMSE** (Hz) và **F0 correlation** giữa real vs synthesized (chỉ trên voiced frames).
  - **V/UV error rate**.
  - **Tone classification accuracy**: chạy một tone classifier **độc lập** (train tách, KHÁC
    classifier dùng trong loss để tránh leakage) trên audio synthesize → so với nhãn.
  - **Tone confusion matrix** theo 6 thanh (định tính, đưa vào paper).
- **Subjective**: MOS + **tone-ABX / tone preference test** với người nghe bản ngữ.
- **Baselines**: HiFi-GAN, Vocos gốc, BigVGAN-v2.
- **Ablation**: Vocos | +C | +C+B | +C+B+A (+D). Mỗi dòng báo cáo đủ metric trên.

---

## 8. Thứ tự triển khai (milestones cho agent)

1. **M0** — Fork Vocos, chạy được train baseline trên 1 dataset tiếng Việt nhỏ, log đầy đủ. ✔ sanity.
2. **M1 (C)** — Thêm F0-weighted STFT loss. Train ngắn, kiểm tra F0 RMSE giảm so với baseline.
3. **M2 (B)** — Thêm CQT tone discriminator. Kiểm tra train ổn định (disc không sụp), log loss cân bằng.
4. **M3 (A)** — Pretrain tone classifier (script riêng) → freeze → thêm tone-FM loss. Đây là bước
   tạo novelty chính; theo dõi tone accuracy trên synthesized audio.
5. **M4** — Eval đầy đủ + ablation table.
6. **M5 (D, optional)** — F0 conditioning nếu còn thời gian.

---

## 9. Bẫy & lưu ý kỹ thuật (đọc kỹ)

- **Sample rate đồng nhất** giữa vocoder, tone classifier, CQT, F0 extractor. Sai là gradient rác.
- **Đừng để `.detach()`** trên đường `audio_hat` đi qua classifier/CQT disc — gradient phải về generator.
- **Classifier & các MelSpec phụ phải `eval()` + `requires_grad=False`**, và **không** nằm trong optimizer.
- **Cân bằng loss**: log mọi term riêng. Tone-FM loss và CQT loss phải cùng bậc độ lớn với các loss
  cũ; nếu lệch, vocoder sẽ bỏ qua hoặc bị nhiễu. Tune coeff theo log thực tế, đừng đoán.
- **Creaky voice (ngã/nặng)**: F0 extractor lỗi nhiều ở đây. Khi tính F0 RMSE/metric, **mask voiced**
  và xử lý riêng đoạn unvoiced/creaky để số liệu công bằng.
- **nnAudio CQT** có thể chậm; precompute không được (cần khả vi cho fake) → để nguyên trên GPU,
  cân nhắc giảm `n_bins` nếu nghẽn.
- **Leakage**: classifier dùng trong loss ≠ classifier dùng để eval. Bắt buộc tách.
- **Reproducibility**: seed, log config, lưu checkpoint classifier kèm code version.

---

## 10. Định nghĩa thanh điệu (mapping nhãn)

```
0 = ngang  (level/mid)         — không dấu
1 = sắc    (high-rising)       — á
2 = huyền  (low-falling)       — à
3 = hỏi    (dipping-rising)    — ả   [thường có creaky nhẹ]
4 = ngã    (rising-glottalized)— ã   [glottalization mạnh]
5 = nặng   (low-falling-short, glottal stop) — ạ  [creaky/glottal]
```

> Nhãn này dùng khi pretrain tone classifier (A.2). Đảm bảo dataset/aligner dùng đúng convention.
