"""Standalone verification for the applied roadmap modules (M1, M7, M9).

No dataset required: builds the real model from config and drives it with
random tensors on CPU. Checks:

  1. M1  BCMAStem is the active stem and adds params vs the decoupled stem.
  2. M7  GatedSkipFusion replaces every concat skip and preserves channel count.
  3. M9  AuxiliaryLoss (SAM/FFT) is wired and produces finite values.
  4. End-to-end forward + backward gives finite gradients (fp32).
  5. The fp16 NaN scenario (saturated SAM cos / large-magnitude FFT) now stays
     finite -- i.e. the NaN fix holds.

Run:  python verify_modules.py
"""
import contextlib
import io
import sys

import torch

from config import find_config_using_name
from models.ecrformer_model import BCMAStem, GatedSkipFusion, DecoupledEncoder
from train import CloudRemovalModel


def _silent(fn, *a, **k):
    """Run fn while swallowing the model's verbose per-forward prints."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*a, **k)


def build(config_name, **overrides):
    cfg = find_config_using_name(config_name)()
    for k, v in overrides.items():
        cfg.net.cfg[k] = v
    # avoid the one-time ~528MB VGG/LPIPS download in this offline check
    cfg.train.aux_loss.perceptual_weight = 0.0
    return _silent(CloudRemovalModel, cfg), cfg


def n_params(model):
    return sum(p.numel() for p in model.net.parameters())


def main():
    torch.manual_seed(0)
    results = []

    # ----- build the full model (M1 + M7 + M9 all on) --------------------
    model, cfg = build("ecrformer")
    net = model.net
    in_ch = sum(cfg.net.cfg["in_chans"])   # SAR(2)+cloudy(13)=15
    out_ch = cfg.net.cfg["out_chans"]      # 13

    # ----- 1. M1: BCMA stem active --------------------------------------
    m1_ok = isinstance(net.stem, BCMAStem)
    results.append(("M1  stem is BCMAStem", m1_ok))

    # M1 should add params vs the plain decoupled stem
    model_no_bcma, _ = build("ecrformer", bcma=False)
    results.append(("M1  decoupled-stem fallback works",
                    isinstance(model_no_bcma.net.stem, DecoupledEncoder)))
    delta_bcma = n_params(model) - n_params(model_no_bcma)
    results.append((f"M1  BCMA adds params (+{delta_bcma})", delta_bcma > 0))

    # ----- 2. M7: gated skips active ------------------------------------
    m7_present = net.skip_gates is not None and len(net.skip_gates) > 0
    m7_type = m7_present and all(isinstance(g, GatedSkipFusion) for g in net.skip_gates)
    results.append(("M7  skip_gates present", bool(m7_present)))
    results.append(("M7  every skip is GatedSkipFusion", bool(m7_type)))

    model_no_gate, _ = build("ecrformer", gated_skip=False)
    results.append(("M7  plain-concat fallback works",
                    model_no_gate.net.skip_gates is None))
    delta_gate = n_params(model) - n_params(model_no_gate)
    results.append((f"M7  gating adds params (+{delta_gate})", delta_gate > 0))

    # M7 channel-preservation: GatedSkipFusion output == concat output channels
    g = net.skip_gates[0]
    dec_ch = g.gate[0].in_channels - g.gate[0].out_channels  # dec = (dec+enc) - enc
    enc_ch = g.gate[0].out_channels
    xd = torch.randn(1, dec_ch, 8, 8)
    xe = torch.randn(1, enc_ch, 8, 8)
    fused = g(xd, xe)
    results.append(("M7  preserves concat channel count",
                    fused.shape[1] == dec_ch + enc_ch))

    # ----- 3. M9: aux losses wired --------------------------------------
    aux = model.aux_loss
    results.append(("M9  aux loss enabled (sam+fft)", aux.enabled))
    results.append(("M9  SAM term present", aux.sam is not None))
    results.append(("M9  FFT term present", aux.fft is not None))

    # ----- 4. end-to-end forward + backward (fp32) ----------------------
    x = torch.randn(2, in_ch, 64, 64)
    pred, (down_projs, up_projs) = _silent(net.forward, x)
    fwd_ok = (pred.shape == (2, out_ch, 64, 64) and torch.isfinite(pred).all().item())
    results.append(("4   forward output shape + finite", fwd_ok))

    target = torch.rand(2, out_ch, 64, 64)
    aux_total, parts = aux(pred, target)
    loss = pred.abs().mean() + aux_total
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    grad_ok = len(grads) > 0 and all(torch.isfinite(gd).all().item() for gd in grads)
    results.append(("4   backward grads finite (fp32)", grad_ok))
    results.append(("4   aux parts finite",
                    all(torch.isfinite(v).all().item() for v in parts.values())))

    # ----- 5. fp16 NaN scenario now safe --------------------------------
    from util.losses import SAMLoss, FFTLoss
    tgt = torch.rand(2, out_ch, 64, 64)
    # SAM: prediction ~identical to target -> cos saturates to 1 (the old crash)
    p_sam = (tgt + 1e-4 * torch.randn_like(tgt)).half().requires_grad_(True)
    l = SAMLoss()(p_sam, tgt.half()); l.backward()
    results.append(("5   SAM fp16 (cos->1) grad finite",
                    torch.isfinite(p_sam.grad).all().item()))
    # FFT: large-magnitude predictions in fp16
    p_fft = (tgt * 30).half().requires_grad_(True)
    l2 = FFTLoss(mode="amplitude")(p_fft, tgt.half()); l2.backward()
    results.append(("5   FFT fp16 (large mag) grad finite",
                    torch.isfinite(p_fft.grad).all().item()))

    # ----- report --------------------------------------------------------
    print("\n" + "=" * 64)
    print("MODULE VERIFICATION (M1 BCMA | M7 gated skips | M9 aux losses)")
    print("=" * 64)
    all_ok = True
    for name, ok in results:
        all_ok &= bool(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name}")
    print("=" * 64)
    print(f"full model params : {n_params(model)/1e6:.4f} M")
    print(f"  vs no-BCMA      : {n_params(model_no_bcma)/1e6:.4f} M  (M1 +{delta_bcma})")
    print(f"  vs no-gated-skip: {n_params(model_no_gate)/1e6:.4f} M  (M7 +{delta_gate})")
    print(f"\nRESULT: {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
