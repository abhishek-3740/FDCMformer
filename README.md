# FDCMformer — ECRformer + M7 + M9

> **Based on [zzaiyan/ECRformer](https://github.com/zzaiyan/ECRformer) (ISPRS 2026).**  
> This fork applies roadmap modifications — M9 (spectral losses), M7 (gated skip connections), and M1 (bi-cross-modal attention fusion) — while staying fully backward-compatible with the original.

---

## What's New

| Module | Description |
|--------|-------------|
| **M9** | Spectral-Angle loss + Fourier amplitude loss + VGG/RemoteCLIP perceptual loss — training-time only, **zero inference cost**. `SAMLoss` directly optimises the SAM metric (clamped arccos for stable gradients); `FFTLoss` enforces frequency-domain consistency (`amplitude` L1 or `focal` Focal-Frequency); `PerceptualLoss` is a frozen VGG-LPIPS feature loss on the RGB bands (B4,B3,B2), optionally swappable for a RemoteCLIP RS backbone |
| **M7** | Gated cross-scale skip connections — `GatedSkipFusion` applies a learnable sigmoid gate `g = σ(Conv1x1([x_dec; x_enc]))` so `out = concat([x_dec, g·x_enc])` suppresses cloud-contaminated encoder skips before fusion. Output channel count is preserved, so the decoder is unchanged (true drop-in); the gate bias is zero-init so it starts at 0.5 (half-pass), near original behaviour. MIMO pyramid supervision was already present in ECRformer (the `up_proj` heads at 1/4, 1/2, 1) and is reused |
| **M1** | Bi-Cross-Modal Attention (BCMA) stem replacing channel-concat fusion — separate SAR/optical stems (SAR stem has a learnable depthwise 5×5 speckle-reduction conv), symmetric cross-covariance cross-attention (`CrossXCA`, channel-wise C×C attention, linear in spatial tokens), and a learned modality gate anchored by a standalone `gate_bias` parameter (init +3.0 → gate ≈ 0.95 self-features at start). Output channels equal `features_start`, so the entire downstream encoder/decoder is unchanged (true drop-in). Fixes shallow concat fusion (B2) |

> **M1 scope:** BCMA is currently applied at the **stem only** (not at every encoder stage). This is a deliberate, cost-conscious first cut — it adds only **~9 K params** (11.4709 M → 11.4802 M for the full config). If E5 shows BCMA helps but under-delivers, the next step is to keep dual streams through the first 1–2 encoder stages and apply `CrossXCA` there too.

To reproduce the original ECRformer, set `sam_weight=fft_weight=perceptual_weight=0.0`, `gated_skip=False`, and `bcma=False`. To recover the M9+M7 model (without M1), set only `bcma=False`.

---

## Files Changed

| File | Change |
|------|--------|
| `util/losses.py` | **NEW** — `SAMLoss`, `FFTLoss`, `PerceptualLoss`, `AuxiliaryLoss` |
| `models/ecrformer_model.py` | Added `GatedSkipFusion` module + `gated_skip` constructor arg; added `CrossXCA` + `BCMAStem` modules + `bcma` constructor arg (M1) |
| `train.py` | Wires `AuxiliaryLoss` into training step; logs `aux_sam`, `aux_fft`, `aux_perceptual` |
| `config/base_config.py` | Added `train.aux_loss` namespace with all M9 knobs |
| `config/ecrformer_config.py` | Added `gated_skip=True` and `bcma=True` to `net.cfg` |
| `requirements.txt` | Added optional `open_clip_torch` (commented) for RemoteCLIP |

---

## Setup

```bash
pip install -r requirements.txt

# CUDA build of PyTorch (example: CUDA 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Optional: RemoteCLIP perceptual backbone
pip install open_clip_torch
```

VGG weights for LPIPS (~528 MB) download automatically on first use.

---

## Data — SEN12MS-CR

1. Download [SEN12MS-CR](https://github.com/PatrickTUM/SEN12MS-CR).
2. Set the root in `config/base_config.py`:
   ```python
   self.dataset.root = r"/path/to/SEN12MS-CR"
   ```
3. *(Optional, faster I/O)* Convert to `.npz`:
   ```bash
   python data/convert_sen12mscr_to_npz.py
   ```
   On Colab you can shard the raw GeoTIFFs to half-precision `.npz` instead. SAR is normalised as `(clip(x, -25, 0) + 25) / 25`, optical as `clip(x, 0, 10000) / 10000`, with the splits:
   - **train** — 36 scenes `{1,6,8,9,15,21,26,39,40,45,58,63,66,75,77,97,100,101,109,110,113,115,117,119,120,121,124,126,128,132,134,141,142,145,147}`
   - **val** — `{17}`
   - **test** — `{31,44,106,123,140}`

Input: 15 channels `[SAR(2), cloudy(13)]` → Output: 13 channels `[target(13)]`, normalised to `[0, 1]` (`dataset.data_range = 1.0`; the SAM/FFT/LPIPS losses assume this range).

---

## Training

```bash
# Full model
python train.py --config ecrformer

# Lightweight model
python train.py --config ecrformer_light

# Useful flags
#  -n NAME      experiment name suffix
#  -g GPU       GPU index (default 0)
#  --no-resume  ignore existing checkpoint, start fresh
```

Monitor with TensorBoard:
```bash
tensorboard --logdir experiments
```
New scalars: `aux_sam`, `aux_fft`, `aux_perceptual` (M1 is purely architectural and adds no new scalar — judge it by the `valid_*` metrics).

**Default schedule:** AdamW `lr=4e-4`, MultiStepLR, max 200 epochs, early-stop patience 10 on `valid_loss`, gradient clipping 0.5. Effective batch size 16 via `batch_size × accumulate_grad_batches` (full: 4×4, light: 8×2). Lower the train batch size if you hit OOM. Auto-resume picks up any checkpoint under `./experiments`; use `--no-resume` to force a clean start.

---

## Evaluation

```bash
python test.py --config ecrformer
python test.py --config ecrformer --ckpt path/to/weights.ckpt
```

Reports: RMSE, MAE, PSNR, SAM, SSIM, LPIPS.

---

## Config Knobs (M9 + M7 + M1)

```python
# config/base_config.py  →  self.train.aux_loss
sam_weight          = 0.05          # spectral-angle loss weight
fft_weight          = 0.10          # Fourier amplitude loss weight
fft_mode            = 'amplitude'   # 'amplitude' | 'focal'
perceptual_weight   = 0.10          # perceptual loss weight
perceptual_backbone = 'lpips'       # 'lpips' | 'remoteclip'
perceptual_net      = 'vgg'         # LPIPS trunk when backbone='lpips'
remoteclip_ckpt     = None          # path to RemoteCLIP weights (remoteclip only)
apply_to_pyramid    = False         # apply M9 losses on up_proj pyramid too

# config/ecrformer_config.py  →  net.cfg
gated_skip          = True          # M7 gated skip connections
bcma                = True          # M1 bi-cross-modal attention stem
```

Total loss: `L = L_base + sam_weight·L_SAM + fft_weight·L_FFT + perceptual_weight·L_perceptual`

---

## Ablation Matrix

| Run | Config | Purpose |
|-----|--------|---------|
| E0 | `sam=fft=perc=0, gated_skip=False` | Baseline (original ECRformer) |
| E1 | E0 + `gated_skip=True` | +M7 only |
| E2 | E1 + `sam=0.05` | +SAM loss |
| E3 | E2 + `fft=0.10` | +FFT loss |
| E4 | E3 + `perceptual=0.10` | Full M7+M9 model |
| E5 | E4 + `bcma=True` | +M1 (BCMA fusion) |

Keep every other hyper-parameter fixed across runs. SAM should drop most with E2/E3; LPIPS should drop most with E4; PSNR/SSIM should rise gradually. For M1 (E5 vs E4) the spec targets — treat as goals, not guarantees, given the stem-only scope and the spring-only ~24 K-sample subset — are `PSNR +0.4..+0.7 dB`, `SAM −0.3..−0.6`, `SSIM +0.003..+0.007`, with the biggest gains expected on thick-cloud samples where SAR structure helps most.

---

## License

Inherits the license of the upstream [ECRformer](https://github.com/zzaiyan/ECRformer) repository.
