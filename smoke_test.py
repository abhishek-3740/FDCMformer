"""CPU smoke test for the FDCMformer training pipeline.

Runs the *real* model (CloudRemovalModel), the *real* dataset (NPZ_Dataset),
and the *real* training/validation steps for a single epoch on a tiny subset
of samples, on CPU. The goal is only to prove the end-to-end pipeline wires
up correctly (data -> fuse_input -> forward -> losses -> backward -> step),
not to train anything useful.

It does NOT touch train.py, so the GPU training path is unchanged.

Usage (from the repo root):
    python smoke_test.py                  # SAM+FFT aux, train + 1 val batch
    python smoke_test.py --no-val         # fastest: skip validation (no LPIPS/VGG download)
    python smoke_test.py --full           # also enable the perceptual (VGG-LPIPS) aux loss
    python smoke_test.py --samples 12 --bs 2

Notes:
  * --no-val avoids the one-time ~528 MB VGG download triggered by the LPIPS
    metric inside validation.
  * The perceptual aux loss is disabled by default (also avoids the VGG
    download); pass --full to exercise it too.
"""

import argparse

import torch
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl

from config import find_config_using_name
from data import find_dataset_using_name
from train import CloudRemovalModel


def parse_args():
    p = argparse.ArgumentParser(description="CPU smoke test for FDCMformer training.")
    p.add_argument("--config", "-c", default="ecrformer",
                   help="config name (default: ecrformer; try ecrformer_light for speed)")
    p.add_argument("--root", default="./dataset",
                   help="dataset root containing <split>/shard_*.npz (default: ./dataset)")
    p.add_argument("--split", default="train", help="split folder name (default: train)")
    p.add_argument("--samples", type=int, default=8,
                   help="total samples to use (split into train/val)")
    p.add_argument("--val-samples", type=int, default=2,
                   help="how many of --samples go to validation")
    p.add_argument("--bs", type=int, default=2, help="train batch size")
    p.add_argument("--no-val", action="store_true",
                   help="skip validation (avoids the LPIPS/VGG ~528MB download)")
    p.add_argument("--full", action="store_true",
                   help="also enable the perceptual (VGG-LPIPS) auxiliary loss")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(42)
    pl.seed_everything(42, workers=True)

    print(f"\n[1/5] Building config '{args.config}' ...")
    config = find_config_using_name(args.config)()
    # CloudRemovalModel reads config.dataset.crop_size for TrainAugment.
    crop = config.dataset.crop_size

    # Keep the smoke test download-free unless --full is requested.
    if not args.full:
        config.train.aux_loss.perceptual_weight = 0.0
        print("      perceptual aux loss DISABLED (use --full to enable VGG-LPIPS).")
    print(f"      aux_loss: sam={config.train.aux_loss.sam_weight} "
          f"fft={config.train.aux_loss.fft_weight} "
          f"perceptual={config.train.aux_loss.perceptual_weight}")
    print(f"      net.cfg: gated_skip={config.net.cfg.get('gated_skip')} "
          f"bcma={config.net.cfg.get('bcma')}")

    print("\n[2/5] Building model on CPU ...")
    model = CloudRemovalModel(config)
    n_params = sum(p.numel() for p in model.net.parameters() if p.requires_grad)
    print(f"      trainable params: {n_params/1e6:.4f} M")

    print(f"\n[3/5] Loading dataset from '{args.root}/{args.split}' ...")
    ds_class = find_dataset_using_name(config.dataset.name)
    full_ds = ds_class(args.root, split=args.split,
                       data_range=config.dataset.data_range, crop_size=crop)
    n_total = min(args.samples, len(full_ds))
    if n_total < 2:
        raise SystemExit("Need at least 2 samples for a train/val split.")
    n_val = min(args.val_samples, n_total - 1) if not args.no_val else 0
    n_train = n_total - n_val
    train_ds = Subset(full_ds, list(range(n_train)))
    print(f"      using {n_total} samples -> {n_train} train / {n_val} val")

    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                              drop_last=True, num_workers=0, pin_memory=False)
    val_loader = None
    if n_val > 0:
        val_ds = Subset(full_ds, list(range(n_train, n_total)))
        val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False,
                                num_workers=0, pin_memory=False)

    print("\n[4/5] Creating CPU trainer (1 epoch) ...")
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        precision=32,
        max_epochs=1,
        gradient_clip_val=0.5,
        accumulate_grad_batches=1,
        logger=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        limit_val_batches=(0 if val_loader is None else 1),
        enable_progress_bar=False,
    )

    print("\n[5/5] Running fit() ...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED: forward + loss + backward + optimizer step ran.")
    print(f"  global_step = {trainer.global_step}")
    if trainer.logged_metrics:
        print("  last logged metrics:")
        for k, v in trainer.logged_metrics.items():
            try:
                print(f"    {k:14s} = {float(v):.5f}")
            except Exception:
                print(f"    {k:14s} = {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
