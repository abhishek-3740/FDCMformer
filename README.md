# FDCMformer — ECRformer + M7 + M9

> **Based on [zzaiyan/ECRformer](https://github.com/zzaiyan/ECRformer) (ISPRS 2026).**  
> This fork applies two roadmap modifications — M9 (spectral losses) and M7 (gated skip connections) — while staying fully backward-compatible with the original.

---

## What's New

| Module | Description |
|--------|-------------|
| **M9** | Spectral-Angle loss + Fourier amplitude loss + VGG/RemoteCLIP perceptual loss — training-time only, **zero inference cost** |
| **M7** | Gated cross-scale skip connections (learnable gate suppresses cloud-contaminated encoder features before fusion); MIMO pyramid supervision was already present in ECRformer and is reused |

To reproduce the original ECRformer, set `sam_weight=fft_weight=perceptual_weight=0.0` and `gated_skip=False`.

---

## Files Changed

| File | Change |
|------|--------|
| `util/losses.py` | **NEW** — `SAMLoss`, `FFTLoss`, `PerceptualLoss`, `AuxiliaryLoss` |
| `models/ecrformer_model.py` | Added `GatedSkipFusion` module + `gated_skip` constructor arg |
| `train.py` | Wires `AuxiliaryLoss` into training step; logs `aux_sam`, `aux_fft`, `aux_perceptual` |
| `config/base_config.py` | Added `train.aux_loss` namespace with all M9 knobs |
| `config/ecrformer_config.py` | Added `gated_skip=True` to `net.cfg` |
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

Input: 15 channels `[SAR(2), cloudy(13)]` → Output: 13 channels `[target(13)]`, normalised to `[0, 1]`.

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
New scalars: `aux_sam`, `aux_fft`, `aux_perceptual`.

---

## Evaluation

```bash
python test.py --config ecrformer
python test.py --config ecrformer --ckpt path/to/weights.ckpt
```

Reports: RMSE, MAE, PSNR, SAM, SSIM, LPIPS.

---

## Config Knobs (M9 + M7)

```python
# config/base_config.py  →  self.train.aux_loss
sam_weight          = 0.05          # spectral-angle loss weight
fft_weight          = 0.10          # Fourier amplitude loss weight
fft_mode            = 'amplitude'   # 'amplitude' | 'focal'
perceptual_weight   = 0.10          # perceptual loss weight
perceptual_backbone = 'lpips'       # 'lpips' | 'remoteclip'
apply_to_pyramid    = False         # apply M9 losses on up_proj pyramid too

# config/ecrformer_config.py  →  net.cfg
gated_skip          = True          # M7 gated skip connections
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
| E4 | E3 + `perceptual=0.10` | Full proposed model |

---

## License

Inherits the license of the upstream [ECRformer](https://github.com/zzaiyan/ECRformer) repository.
