"""Auxiliary training losses for NEXT-CR (M9).

This module implements the three differentiable training-loss terms proposed in
the M9 modification of the NEXT-CR roadmap:

    * ``SAMLoss``        -- Spectral-Angle-Mapper loss (directly optimises SAM).
    * ``FFTLoss``        -- Fourier amplitude / focal-frequency consistency loss.
    * ``PerceptualLoss`` -- frozen feature-space perceptual loss. Defaults to the
                            VGG-LPIPS backbone that already ships with the repo
                            (no extra download), with an optional hook for a
                            remote-sensing foundation model (RemoteCLIP).

All losses operate on tensors shaped ``[B, C, H, W]`` with values in the data
range used by the rest of the pipeline (SEN12MS-CR is normalised to ``[0, 1]``,
``config.dataset.data_range == 1.0``).

The losses are deliberately self-contained and add **zero** inference cost: they
are only ever called inside ``training_step``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Sentinel-2 RGB channel indices (B4, B3, B2) inside the 13-band tensor.
DEFAULT_RGB_INDEX = (3, 2, 1)


def _load_lpips_loss():
    """Lazily import the VGG-LPIPS module (keeps SAM/FFT free of the lpips dep)."""
    try:
        from .lpips_loss import LPIPSLoss
    except ImportError:
        from lpips_loss import LPIPSLoss
    return LPIPSLoss


# ---------------------------------------------------------------------------
# M9.1 -- Spectral-Angle-Mapper loss
# ---------------------------------------------------------------------------

class SAMLoss(nn.Module):
    """Differentiable Spectral-Angle-Mapper loss.

    For every pixel the spectral vectors of the prediction and target (across
    the channel dimension) are compared by the angle between them:

        SAM(p, t) = arccos( <p, t> / (||p|| * ||t||) )

    The mean angle (in radians) over all pixels/batch is returned. SAM is one of
    the headline metrics for cloud removal yet the baseline never optimises it
    directly -- this term closes that gap.

    Args:
        eps: numerical-stability constant.
        clamp_eps: the cosine is clamped to ``[-1 + clamp_eps, 1 - clamp_eps]``
            before ``arccos`` to keep the gradient finite (the derivative of
            ``arccos`` diverges at ``±1``).
    """

    def __init__(self, eps: float = 1e-6, clamp_eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        # NOTE: clamp_eps MUST be well above fp16 resolution (~1e-3 near 1.0),
        # otherwise under autocast `1 - clamp_eps` rounds back to exactly 1.0,
        # the clamp becomes a no-op, and acos'(±1) = -1/sqrt(1-cos^2) = -inf
        # blows up the gradient. 1e-4 caps |grad| at ~1/sqrt(2e-4) ~= 70.
        self.clamp_eps = clamp_eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Always evaluate in fp32: acos / sqrt are numerically fragile in fp16.
        pred = pred.float()
        target = target.float()
        # dot product over the spectral (channel) dimension
        dot = torch.sum(pred * target, dim=1)
        pred_norm = torch.sqrt(torch.sum(pred * pred, dim=1) + self.eps)
        target_norm = torch.sqrt(torch.sum(target * target, dim=1) + self.eps)
        cos = dot / (pred_norm * target_norm + self.eps)
        cos = torch.clamp(cos, -1.0 + self.clamp_eps, 1.0 - self.clamp_eps)
        angle = torch.acos(cos)  # radians, shape [B, H, W]
        return angle.mean()


# ---------------------------------------------------------------------------
# M9.2 -- Fourier amplitude / focal-frequency loss
# ---------------------------------------------------------------------------

class FFTLoss(nn.Module):
    """Frequency-domain amplitude-consistency loss.

    Clouds are dominated by a strong low-frequency signature while scene texture
    lives in the mid/high bands. Matching the Fourier amplitude spectra of the
    prediction and target gives an explicit frequency-domain training signal.

    Two variants are supported:
        * ``mode='amplitude'`` (default): L1 between the magnitude spectra
          ``|| |FFT(pred)| - |FFT(target)| ||_1``.
        * ``mode='focal'``: the Focal Frequency Loss (Jiang et al., ICCV 2021),
          which additionally compares the phase and adaptively re-weights hard
          frequencies by the squared spectral distance.
    """

    def __init__(self, mode: str = "amplitude", alpha: float = 1.0,
                 eps: float = 1e-8):
        super().__init__()
        assert mode in ("amplitude", "focal"), f"Unsupported FFT loss mode: {mode}"
        self.mode = mode
        self.alpha = alpha
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # FFT is ill-defined / overflow-prone in fp16 (amplitudes can exceed the
        # fp16 max of 65504 -> inf). Always compute in fp32.
        pred = pred.float()
        target = target.float()
        # 2-D FFT over the spatial dims; complex output [B, C, H, W//2 + 1]
        pred_fft = torch.fft.rfft2(pred, norm="ortho")
        target_fft = torch.fft.rfft2(target, norm="ortho")

        if self.mode == "amplitude":
            # Compute the magnitude manually with an eps inside the sqrt.
            # torch.abs() of a complex tensor has a 1/|z| term in its backward
            # pass that becomes inf/NaN at near-zero frequency bins; the +eps
            # below keeps the gradient finite.
            pred_amp = torch.sqrt(
                pred_fft.real ** 2 + pred_fft.imag ** 2 + self.eps)
            target_amp = torch.sqrt(
                target_fft.real ** 2 + target_fft.imag ** 2 + self.eps)
            return F.l1_loss(pred_amp, target_amp)

        # focal frequency loss: distance in the complex plane, adaptively weighted
        diff = pred_fft - target_fft
        sq_dist = diff.real ** 2 + diff.imag ** 2          # |pred - target|^2
        weight = sq_dist.detach() ** self.alpha            # focal weight (no grad)
        weight = weight / (weight.max() + 1e-8)
        return (weight * sq_dist).mean()


# ---------------------------------------------------------------------------
# M9.3 -- Foundation-model / perceptual loss
# ---------------------------------------------------------------------------

class PerceptualLoss(nn.Module):
    """Frozen feature-space perceptual loss.

    By default this wraps the VGG-LPIPS network already used elsewhere in the
    repository (so it is differentiable and needs no extra setup). For the
    remote-sensing domain a foundation-model backbone is preferable; set
    ``backbone='remoteclip'`` to use RemoteCLIP features instead. The RemoteCLIP
    path is optional and loaded lazily so that the default training run never
    depends on an external download.

    Args:
        backbone: ``'lpips'`` (default) or ``'remoteclip'``.
        rgb_index: indices of the R,G,B channels inside the multispectral tensor
            (Sentinel-2 B4,B3,B2 -> ``(3, 2, 1)``).
        net: LPIPS feature extractor when ``backbone='lpips'`` ('vgg'/'alex'/'squeeze').
    """

    def __init__(self, backbone: str = "lpips", rgb_index=DEFAULT_RGB_INDEX,
                 net: str = "vgg", remoteclip_ckpt: str | None = None):
        super().__init__()
        self.backbone = backbone.lower()
        self.rgb_index = tuple(rgb_index)

        if self.backbone == "lpips":
            LPIPSLoss = _load_lpips_loss()
            self.model = LPIPSLoss(RGB_index=self.rgb_index, net=net)
        elif self.backbone == "remoteclip":
            self.model = _RemoteCLIPPerceptual(
                rgb_index=self.rgb_index, ckpt_path=remoteclip_ckpt)
        else:
            raise ValueError(f"Unknown perceptual backbone: {backbone}")

        # the perceptual backbone is always frozen
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.backbone == "lpips":
            return self.model(pred, target, reduction="mean")
        return self.model(pred, target)


class _RemoteCLIPPerceptual(nn.Module):
    """Optional RemoteCLIP-based perceptual loss (lazy, best-effort).

    Requires ``open_clip_torch`` and the RemoteCLIP weights. Kept isolated so
    that importing :mod:`util.losses` never fails when those are absent.
    """

    # ImageNet/CLIP normalisation statistics
    _MEAN = (0.48145466, 0.4578275, 0.40821073)
    _STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, rgb_index=DEFAULT_RGB_INDEX, ckpt_path: str | None = None,
                 model_name: str = "ViT-B-32"):
        super().__init__()
        try:
            import open_clip
        except ImportError as exc:  # pragma: no cover - optional path
            raise ImportError(
                "RemoteCLIP perceptual loss requires `open_clip_torch`. "
                "Install it or use backbone='lpips'."
            ) from exc

        self.rgb_index = tuple(rgb_index)
        model, _, _ = open_clip.create_model_and_transforms(model_name)
        if ckpt_path is not None:
            state = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state, strict=False)
        self.visual = model.visual.eval()
        self.register_buffer("mean", torch.tensor(self._MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(self._STD).view(1, 3, 1, 1))

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, self.rgb_index, :, :]
        x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        fp = self.visual(self._prep(pred))
        ft = self.visual(self._prep(target))
        return F.l1_loss(fp, ft)


# ---------------------------------------------------------------------------
# Convenience container
# ---------------------------------------------------------------------------

class AuxiliaryLoss(nn.Module):
    """Bundles the M9 loss terms and applies their configured weights.

    Only the terms with a strictly positive weight are instantiated, so disabling
    a term (weight ``0``) costs nothing. ``forward`` returns ``(total, parts)``
    where ``parts`` is a dict of the individual *unweighted* values for logging.
    """

    def __init__(self,
                 sam_weight: float = 0.0,
                 fft_weight: float = 0.0,
                 perceptual_weight: float = 0.0,
                 fft_mode: str = "amplitude",
                 perceptual_backbone: str = "lpips",
                 perceptual_net: str = "vgg",
                 rgb_index=DEFAULT_RGB_INDEX,
                 remoteclip_ckpt: str | None = None):
        super().__init__()
        self.sam_weight = float(sam_weight)
        self.fft_weight = float(fft_weight)
        self.perceptual_weight = float(perceptual_weight)

        self.sam = SAMLoss() if self.sam_weight > 0 else None
        self.fft = FFTLoss(mode=fft_mode) if self.fft_weight > 0 else None
        self.perceptual = (
            PerceptualLoss(backbone=perceptual_backbone, rgb_index=rgb_index,
                           net=perceptual_net, remoteclip_ckpt=remoteclip_ckpt)
            if self.perceptual_weight > 0 else None
        )

    @property
    def enabled(self) -> bool:
        return any(m is not None for m in (self.sam, self.fft, self.perceptual))

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        total = pred.new_zeros(())
        parts: dict[str, torch.Tensor] = {}

        if self.sam is not None:
            v = self.sam(pred, target)
            parts["sam"] = v.detach()
            total = total + self.sam_weight * v
        if self.fft is not None:
            v = self.fft(pred, target)
            parts["fft"] = v.detach()
            total = total + self.fft_weight * v
        if self.perceptual is not None:
            v = self.perceptual(pred, target)
            parts["perceptual"] = v.detach()
            total = total + self.perceptual_weight * v

        return total, parts
