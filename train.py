import os
import json
from datetime import timedelta
from argparse import Namespace as _Namespace
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import ArgumentParser, Namespace

from util.pytorch_ssim import SSIM
from util.util import count_parameters, initialize_weights, compute_metric
from util.losses import AuxiliaryLoss
from util.augment import TestAugment, TrainAugment
from util.checkpoint import find_latest_checkpoint
from util.data_split import build_train_valid_datasets

from models import find_model_using_name
from data import find_dataset_using_name
from config import find_config_using_name


class CloudRemovalModel(pl.LightningModule):
    """PyTorch Lightning module for cloud removal with ECRformer.

    Supports multi-scale projection loss (down_proj + up_proj).
    """

    def __init__(self, config: Namespace):
        super().__init__()
        self.config = config
        self.input = config.net.input
        self.output = config.net.output

        net_class = find_model_using_name(config.net.name)
        self.net = net_class(**config.net.cfg)

        self.SSIM = SSIM()
        self.downsample = nn.AvgPool2d(2, 2)
        def loss_SSIM(x1, x2): return 1 - self.SSIM(x1, x2)
        self.loss_fn = [nn.L1Loss(), loss_SSIM]

        self.loss_weight = list(config.train.loss_weight[:2])
        if len(self.loss_weight) != 2:
            raise ValueError(
                f'loss_weight must provide at least two entries, got {len(config.train.loss_weight)}'
            )

        self.proj_weight = config.train.proj_weight

        self.lr = config.train.lr
        self.train_augment = TrainAugment(crop_size=config.dataset.crop_size)
        # Don't pickle the raw config object into the checkpoint: under
        # torch>=2.6 (weights_only=True) that breaks resume. The model is
        # always rebuilt from config before fit(), and a human-readable
        # config_snapshot.json is written next to the logs.
        self.save_hyperparameters(ignore=['config'])

        initialize_weights(self.net)

        down_weight, up_weight = self.proj_weight
        if down_weight == 0.:
            self.net.down_proj.requires_grad_(False)
        if up_weight == 0.:
            self.net.up_proj.requires_grad_(False)

        # M9: auxiliary losses (SAM + FFT + perceptual). Frozen backbones, so
        # they add zero inference cost and no trainable parameters.
        aux_cfg = getattr(config.train, 'aux_loss', None)
        self.aux_apply_to_pyramid = False
        if aux_cfg is not None:
            self.aux_loss = AuxiliaryLoss(
                sam_weight=getattr(aux_cfg, 'sam_weight', 0.0),
                fft_weight=getattr(aux_cfg, 'fft_weight', 0.0),
                perceptual_weight=getattr(aux_cfg, 'perceptual_weight', 0.0),
                fft_mode=getattr(aux_cfg, 'fft_mode', 'amplitude'),
                perceptual_backbone=getattr(aux_cfg, 'perceptual_backbone', 'lpips'),
                perceptual_net=getattr(aux_cfg, 'perceptual_net', 'vgg'),
                remoteclip_ckpt=getattr(aux_cfg, 'remoteclip_ckpt', None),
            )
            self.aux_apply_to_pyramid = bool(
                getattr(aux_cfg, 'apply_to_pyramid', False))
        else:
            self.aux_loss = AuxiliaryLoss()  # all weights 0 -> disabled

    def forward(self, x, *args, **kwargs):
        return self.net(x, *args, **kwargs)

    @torch.no_grad()
    def fuse_input(self, batch):
        """Fuse input channels and prepare target tensor."""
        if self.training:
            batch = self.train_augment.augment(batch)
        target = torch.cat([batch[key] for key in self.output], dim=1)
        merged = torch.cat([batch[key] for key in self.input], dim=1)
        return batch, merged, target

    def training_step(self, batch, batch_idx):
        batch, merged, target = self.fuse_input(batch)
        pred, projs = self.forward(merged)

        down_projs, up_projs = projs

        # Multi-Scale Feature Regularization
        down_loss_sum, up_loss_sum = 0, 0
        mid_target = target
        for down_proj, up_proj in zip(down_projs, up_projs[::-1]):
            down_loss = [fn(down_proj, mid_target) for fn in self.loss_fn]
            down_loss = sum(l * w for l, w in zip(down_loss, [0.1, 0.9]))
            down_loss_sum = down_loss_sum + down_loss

            up_loss = [fn(up_proj, mid_target) for fn in self.loss_fn]
            up_loss = sum(l * w for l, w in zip(up_loss, [0.9, 0.1]))
            up_loss_sum = up_loss_sum + up_loss

            mid_target = self.downsample(mid_target)

        down_loss = down_loss_sum / (len(down_projs) + 1e-3)
        up_loss = up_loss_sum / (len(up_projs) + 1e-3)

        # Main loss
        loss_list = [fn(pred, target) for fn in self.loss_fn]
        loss = sum(l * w for l, w in zip(loss_list, self.loss_weight))

        # Combined loss
        down_weight, up_weight = self.proj_weight
        loss = loss * (1 - down_weight - up_weight) + \
            down_loss * down_weight + up_loss * up_weight

        # M9: auxiliary losses (SAM + FFT + perceptual) on the full-res prediction.
        # These terms (acos, complex-FFT magnitude, VGG features) are numerically
        # fragile in fp16, so under mixed precision they are evaluated in fp32 with
        # autocast disabled to avoid inf/NaN gradients.
        if self.aux_loss.enabled:
            with torch.autocast(device_type=pred.device.type, enabled=False):
                aux_total, aux_parts = self.aux_loss(pred.float(), target.float())

                # Optionally also apply on the multi-scale up_proj pyramid (M7 x M9)
                if self.aux_apply_to_pyramid:
                    mid_target = target
                    for up_proj in up_projs[::-1]:
                        a_total, _ = self.aux_loss(up_proj.float(), mid_target.float())
                        aux_total = aux_total + a_total
                        mid_target = self.downsample(mid_target)

            loss = loss + aux_total.to(loss.dtype)
            for name, val in aux_parts.items():
                self.log(f'aux_{name}', val, prog_bar=False)

        self.log('train_MAE', loss_list[0], prog_bar=True)
        self.log('train_SSIM', 1 - loss_list[1], prog_bar=True)
        self.log('down_proj', down_loss, prog_bar=True)
        self.log('up_proj', up_loss, prog_bar=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        batch, merged, target = self.fuse_input(batch)
        pred, projs = self.forward(merged)

        metrics = compute_metric(pred, target, size_average=True)
        self.log('valid_RMSE', metrics['RMSE'], prog_bar=False, on_epoch=True)
        self.log('valid_MAE', metrics['MAE'], prog_bar=True, on_epoch=True)
        self.log('valid_PSNR', metrics['PSNR'], prog_bar=False, on_epoch=True)
        self.log('valid_SAM', metrics['SAM'], prog_bar=False, on_epoch=True)
        self.log('valid_SSIM', metrics['SSIM'], prog_bar=True, on_epoch=True)
        self.log('valid_LPIPS', metrics['LPIPS'], prog_bar=False, on_epoch=True)
        valid_loss = metrics['MAE'] * self.loss_weight[0] + \
            (1 - metrics['SSIM']) * self.loss_weight[1]
        self.log('valid_loss', valid_loss, prog_bar=False, on_epoch=True)
        if len(self.trainer.optimizers) > 0:
            self.log('learning_rate',
                     self.trainer.optimizers[0].param_groups[0]['lr'],
                     prog_bar=False, on_epoch=True)

    @torch.no_grad()
    def predict_step(self, batch, batch_idx):
        batch, merged, target = self.fuse_input(batch)
        pred_result = self.forward(merged)
        return batch, pred_result

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=self.lr, weight_decay=1e-3)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode='min', factor=0.5, patience=10)
        # return {
        #     'optimizer': optimizer,
        #     'lr_scheduler': {
        #         'scheduler': scheduler,
        #         'monitor': 'valid_loss',
        #         'interval': 'epoch',
        #         'frequency': 1,
        #     },
        # }

        milestones = list(getattr(self.config.train, 'lr_milestones',
                                  [120, 150, 170, 180, 190, 200]))
        gamma = float(getattr(self.config.train, 'lr_gamma', 0.5))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=gamma)
        return [optimizer], [scheduler]


# ---------------------------------------------------------------------------
# Training Entry Point
# ---------------------------------------------------------------------------

def _to_plain(obj):
    """Recursively convert Namespaces/containers into JSON-friendly types."""
    if isinstance(obj, _Namespace):
        return {k: _to_plain(v) for k, v in vars(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def dump_config_snapshot(config, log_dir):
    """Write the resolved config to <log_dir>/config_snapshot.json (rank 0 only)."""
    try:
        if log_dir is None:
            return
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, 'config_snapshot.json')
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(_to_plain(config), fh, indent=2, ensure_ascii=False)
        print(f"Config snapshot saved: {path}")
    except Exception as exc:  # never let logging break training
        print(f"[warn] could not write config snapshot: {exc}")


def main(config):
    torch.set_float32_matmul_precision("highest")
    pl.seed_everything(config.seed)

    print("\nBuilding model...")
    model = CloudRemovalModel(config)
    print(f"Model class: {model.net.__class__.__name__}")
    count_parameters(model)

    print("\nLoading dataset...")
    dataset_class = find_dataset_using_name(config.dataset.name)
    train_dataset, valid_dataset, split_info = build_train_valid_datasets(
        config, dataset_class)
    print(f"Training split: {split_info['train']}")
    print(f"Validation split: {split_info['valid']}")

    num_workers = config.train.num_workers
    persistent = num_workers > 0
    train_loader = DataLoader(
        train_dataset, batch_size=config.train.train_bs, drop_last=True,
        shuffle=True, num_workers=num_workers,
        pin_memory=True, persistent_workers=persistent)
    valid_loader = DataLoader(
        valid_dataset, batch_size=config.train.valid_bs,
        shuffle=False, num_workers=num_workers,
        pin_memory=True, persistent_workers=persistent)

    # --- Checkpointing strategy -----------------------------------------
    # 1) Keep the top-k best models (by valid_loss) with a descriptive name.
    save_top_k = int(getattr(config.train, 'save_top_k', 1))
    best_ckpt = ModelCheckpoint(
        monitor='valid_loss', mode='min', verbose=False,
        save_top_k=save_top_k, save_last=False,
        filename='best-{epoch:03d}-{valid_loss:.4f}',
        auto_insert_metric_name=True)
    # 2) A rolling `last.ckpt` for resume, refreshed every epoch AND every N
    #    minutes of wall-clock (so a Kaggle timeout mid-epoch loses little).
    every_n_min = float(getattr(config.train, 'ckpt_every_n_min', 0) or 0)
    resume_ckpt_kwargs = dict(save_top_k=0, save_last=True, verbose=False)
    if every_n_min > 0:
        resume_ckpt_kwargs['train_time_interval'] = timedelta(minutes=every_n_min)
    resume_ckpt = ModelCheckpoint(**resume_ckpt_kwargs)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    early_stop_callback = EarlyStopping(
        monitor='valid_loss', patience=config.train.early_stop,
        verbose=True, mode='min')
    callbacks = [best_ckpt, resume_ckpt, lr_monitor, early_stop_callback]

    log_name = config_name
    if config.name:
        log_name = f"{log_name}_{config.name}"

    save_dir = config.train.save_dir

    ckpt_path = config.train.ckpt_path
    resume_version = None

    if ckpt_path is None and not getattr(config, 'no_resume', False):
        auto_ckpt_path, version_num = find_latest_checkpoint(
            save_dir, log_name)
        if auto_ckpt_path is not None:
            ckpt_path = auto_ckpt_path
            resume_version = version_num
            print(f"Auto-resume checkpoint: {ckpt_path}")
        else:
            print("No checkpoint found. Training from scratch.")
    elif getattr(config, 'no_resume', False):
        print("Auto-resume is disabled.")
        ckpt_path = None

    if resume_version is not None:
        tb_logger = TensorBoardLogger(
            save_dir=save_dir, name=log_name, version=resume_version)
    else:
        tb_logger = TensorBoardLogger(save_dir=save_dir, name=log_name)

    # Persist a human-readable snapshot of the resolved config next to the logs.
    dump_config_snapshot(config, tb_logger.log_dir)

    print("\nCreating trainer...")
    # Multi-GPU (e.g. Kaggle 2x T4): config.train.gpu may be an int count or a
    # list of indices; config.train.strategy selects DDP ('ddp' for scripts,
    # 'ddp_notebook' inside a Kaggle notebook). Single-GPU stays the default.
    devices = config.train.gpu
    strategy = getattr(config.train, 'strategy', 'auto')
    max_time = getattr(config.train, 'max_time', None)
    trainer = pl.Trainer(
        max_epochs=config.train.max_epoch,
        max_time=max_time,
        gradient_clip_val=.5,
        callbacks=callbacks,
        logger=[tb_logger],
        devices=devices,
        strategy=strategy,
        **config.optim.__dict__,
    )

    if max_time is not None:
        print(f"Wall-clock budget (max_time): {max_time} "
              "(stops gracefully + checkpoints before this).")
    print("\nStarting training...")
    print(f"Experiment directory: {os.path.join(save_dir, log_name)}")
    trainer.fit(model, train_dataloaders=train_loader,
                val_dataloaders=valid_loader, ckpt_path=ckpt_path)
    print("\nTraining finished.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', '-c', type=str,
                        default='ecrformer', help="Config name")
    parser.add_argument('--name', '-n', type=str, default=None)
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--devices', type=int, default=None,
                        help="Number of GPUs (e.g. 2 for Kaggle 2x T4). "
                             "Overrides --gpu when set; enables DDP.")
    parser.add_argument('--strategy', type=str, default=None,
                        help="Lightning strategy: 'ddp' (script) or "
                             "'ddp_notebook' (Kaggle notebook). Default 'auto'.")
    parser.add_argument('--precision', type=str, default=None,
                        help="Trainer precision, e.g. '16-mixed' for T4, '32'.")
    parser.add_argument('--no-resume', action='store_true',
                        help="Disable automatic checkpoint resume")
    args = parser.parse_args()

    config_name = args.config
    print(f"Config: {config_name}")
    config_class = find_config_using_name(config_name)
    config = config_class()
    config.name = args.name

    if args.devices is not None and args.devices > 1:
        config.train.gpu = args.devices
        # Default to DDP when going multi-GPU unless config/CLI overrides it.
        if args.strategy is None and getattr(config.train, 'strategy', 'auto') == 'auto':
            config.train.strategy = 'ddp'
    else:
        config.train.gpu = [args.gpu]

    if args.strategy is not None:
        config.train.strategy = args.strategy
    if args.precision is not None:
        config.optim.precision = args.precision

    config.no_resume = args.no_resume

    main(config)
