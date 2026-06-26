# FDCMformer — ECRformer + M1 + M7 + M9

> **Based on [zzaiyan/ECRformer](https://github.com/zzaiyan/ECRformer) (ISPRS 2026).**  
> This fork applies roadmap modifications — M9 (spectral losses), M7 (gated skip connections), and M1 (bi-cross-modal attention fusion) — while staying fully backward-compatible with the original.

---

## What's New

| Module | Description |
|--------|-------------|
| **M9** | Spectral-Angle loss + Fourier amplitude loss + VGG/RemoteCLIP perceptual loss — training-time only, **zero inference cost**. `SAMLoss` directly optimises the SAM metric (fp32 clamped arccos with an **fp16-safe clamp margin** for stable gradients); `FFTLoss` enforces frequency-domain consistency (`amplitude` L1, computed via a **guarded magnitude** `sqrt(re²+im²+ε)` instead of `complex.abs()`, or `focal` Focal-Frequency); `PerceptualLoss` is a frozen VGG-LPIPS feature loss on the RGB bands (B4,B3,B2), optionally swappable for a RemoteCLIP RS backbone. All three terms are evaluated in **fp32 with autocast disabled** so they are safe under mixed precision (see [Mixed precision & numerical stability](#mixed-precision--numerical-stability)) |
| **M7** | Gated cross-scale skip connections — `GatedSkipFusion` applies a learnable sigmoid gate `g = σ(Conv1x1([x_dec; x_enc]))` so `out = concat([x_dec, g·x_enc])` suppresses cloud-contaminated encoder skips before fusion. Output channel count is preserved, so the decoder is unchanged (true drop-in); the gate bias is zero-init so it starts at 0.5 (half-pass), near original behaviour. MIMO pyramid supervision was already present in ECRformer (the `up_proj` heads at 1/4, 1/2, 1) and is reused |
| **M1** | Bi-Cross-Modal Attention (BCMA) stem replacing channel-concat fusion — separate SAR/optical stems (SAR stem has a learnable depthwise 5×5 speckle-reduction conv), symmetric cross-covariance cross-attention (`CrossXCA`, channel-wise C×C attention, linear in spatial tokens), and a learned modality gate anchored by a standalone `gate_bias` parameter (init +3.0 → gate ≈ 0.95 self-features at start). Output channels equal `features_start`, so the entire downstream encoder/decoder is unchanged (true drop-in). Fixes shallow concat fusion (B2) |

> **M1 scope:** BCMA is currently applied at the **stem only** (not at every encoder stage). This is a deliberate, cost-conscious first cut — it adds only **~9 K params** (11.4709 M → 11.4802 M for the full config). If E5 shows BCMA helps but under-delivers, the next step is to keep dual streams through the first 1–2 encoder stages and apply `CrossXCA` there too.

To reproduce the original ECRformer, set `sam_weight=fft_weight=perceptual_weight=0.0`, `gated_skip=False`, and `bcma=False`. To recover the M9+M7 model (without M1), set only `bcma=False`.

---

## Files Changed

| File | Change |
|------|--------|
| `util/losses.py` | **NEW** — `SAMLoss`, `FFTLoss`, `PerceptualLoss`, `AuxiliaryLoss`. SAM/FFT compute in **fp32** with an fp16-safe clamp margin and a guarded FFT magnitude (no `complex.abs()` backward blow-up) |
| `models/ecrformer_model.py` | Added `GatedSkipFusion` module + `gated_skip` constructor arg; added `CrossXCA` + `BCMAStem` modules + `bcma` constructor arg (M1) |
| `train.py` | Wires `AuxiliaryLoss` into training step; logs `aux_sam`, `aux_fft`, `aux_perceptual`. Aux losses run under **`autocast(enabled=False)`** (fp32) for mixed-precision stability |
| `config/base_config.py` | Added `train.aux_loss` namespace with all M9 knobs |
| `config/ecrformer_config.py` | Added `gated_skip=True` and `bcma=True` to `net.cfg` |
| `config/ecrformer_kaggle_config.py` | **NEW** — dual-T4 Kaggle preset (auto-detected dataset roots, `16-mixed` precision, DDP, 80-epoch schedule, graceful 8h30m stop + rolling checkpoint) |
| `kaggle_train.ipynb` | **NEW** — end-to-end Kaggle notebook (copy code → install deps → resume → sanity-check → train) |
| `verify_modules.py` | **NEW** — offline test asserting M1/M7/M9 are wired and the fp16 NaN path is finite (no dataset needed) |
| `smoke_test.py` | **NEW** — CPU end-to-end pipeline test on a tiny sample subset |
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

### Training on Kaggle (dual T4)

`kaggle_train.ipynb` + `config/ecrformer_kaggle_config.py` provide a turnkey dual-T4 preset: `16-mixed` precision, DDP across both GPUs, an 80-epoch schedule, and auto-detected dataset roots under `/kaggle/input`. To survive Kaggle's 9 h session cap it stops **cleanly and checkpoints at 8h30m** (`max_time`), refreshes a rolling `last.ckpt` every 30 min, and keeps the top-3 best models.

```bash
python train.py -c ecrformer_kaggle --devices 2 --strategy ddp
```

**Resuming across sessions:** when a version times out, its `/kaggle/working/experiments` is saved as the notebook Output. Start a new version, **attach that previous output as an input**, and Run All — the notebook copies the checkpoints back and `train.py` auto-resumes from `last.ckpt` (epoch + LR + optimizer state restored). For a guaranteed from-scratch run, simply **do not attach any previous output** (or pass `--no-resume`).

---

## Mixed precision & numerical stability

The M9 auxiliary losses are numerically fragile under fp16 (`16-mixed`) and were the cause of a mid-training NaN blow-up. They are now hardened:

- **`SAMLoss`** is computed in **fp32** with a clamp margin (`clamp_eps=1e-4`) sized for fp16. The earlier `1e-6` margin rounded to `0` in fp16, so `1 - clamp_eps` became exactly `1.0`, the clamp was a no-op, and `acos'(±1) = -1/√(1-cos²) = -∞` exploded the gradient once predictions aligned with the target (`cos → 1`). The `1e-4` margin caps `|grad|` at ≈ 70.
- **`FFTLoss`** is computed in **fp32**, and the amplitude uses a **guarded magnitude** `sqrt(re²+im²+ε)` instead of `complex.abs()`, whose backward has a `1/|z|` term that becomes `NaN` at near-zero frequency bins. fp32 also avoids fp16 FFT-amplitude overflow.
- In `train.py` the whole aux-loss block runs under **`torch.autocast(..., enabled=False)`** with `.float()` inputs, so SAM/FFT/LPIPS all evaluate in fp32 even under mixed precision.

> Note on gradient clipping: `gradient_clip_val=0.5` clips by global **norm**. A single `inf`/`NaN` gradient makes the norm `NaN`, which then poisons *every* parameter — so one bad aux-loss step is enough to permanently corrupt the run. If you ever still see NaNs, fall back to full precision with `--precision 32`.

---

## Verifying the build

```bash
# Offline, no dataset needed: asserts M1/M7/M9 are wired and the fp16 NaN path is finite
python verify_modules.py

# CPU end-to-end pipeline test on a tiny sample subset (needs a few .npz shards)
python smoke_test.py --no-val
```

`verify_modules.py` confirms the BCMA stem is active (and adds ~9 K params), every skip is a `GatedSkipFusion` with preserved channel count, the SAM/FFT aux terms are enabled, and that forward/backward — including the fp16 SAM-saturation and large-magnitude FFT cases — produce finite gradients.

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
