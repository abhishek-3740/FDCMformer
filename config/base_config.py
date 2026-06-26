from argparse import Namespace
from typing import Any


class BaseConfig(Namespace):
    NUM_CHANS = {
        'SAR': 2,
        'cloudy': 13,
        'target': 13,
    }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.seed = 42

        # 数据集
        self.dataset = Namespace(
            name='npz',
            root=r"./data/npz",  # folder containing train.npz, val.npz, test.npz
            # Extra roots searched ONLY for the train split (e.g. a second
            # Kaggle dataset holding additional train shards). val/test use
            # `root` only. Each entry may be a flat dir of shard_*.npz.
            train_extra_roots=[],
            split=["train", "val", "test"],
            train_ratio=0.8, # enabled when split only contains one element
            data_range=1.0,
            crop_size=128,
        )

        # 训练
        self.train = Namespace(
            max_epoch=200,
            early_stop=10,
            lr=4e-4,
            # LR schedule (MultiStepLR). Scale milestones to your epoch budget.
            lr_milestones=[120, 150, 170, 180, 190, 200],
            lr_gamma=0.5,
            loss_weight=[0.9, 0.1],
            proj_weight=[0., 0.],
            train_bs=8,
            valid_bs=16,
            num_workers=8,
            gpu=[0],                  # int count (multi-GPU) or list of indices
            strategy='auto',          # 'auto' | 'ddp' | 'ddp_notebook'
            # Checkpointing / resume robustness.
            save_top_k=1,             # number of best (valid_loss) checkpoints kept
            ckpt_every_n_min=0,       # also refresh last.ckpt every N minutes (0=off)
            max_time=None,            # wall-clock cap, e.g. '00:08:30:00' (DD:HH:MM:SS)
            ckpt_path=None,
            save_dir='./experiments',
            # M9: auxiliary training losses (zero inference cost).
            # Set a weight to 0 to disable that term.
            aux_loss=Namespace(
                sam_weight=0.05,          # spectral-angle loss
                fft_weight=0.1,           # Fourier amplitude consistency
                fft_mode='amplitude',     # 'amplitude' | 'focal'
                perceptual_weight=0.1,    # frozen feature-space perceptual loss
                perceptual_backbone='lpips',   # 'lpips' | 'remoteclip'
                perceptual_net='vgg',          # LPIPS backbone when 'lpips'
                remoteclip_ckpt=None,          # path to RemoteCLIP weights
                # apply aux losses on the multi-scale up_proj pyramid too (M7 x M9)
                apply_to_pyramid=False,
            ),
        )

        # 训练优化（传递给 pl.Trainer）
        self.optim = Namespace(
            accelerator='auto',
            precision=32,
            log_every_n_steps=50,
        )

        # 网络
        self.net = Namespace(
            name='ecrformer',
            input=['SAR', 'cloudy'],
            output=['target'],
            cfg=dict(),
        )

        self.net.cfg['in_chans'] = [self.NUM_CHANS[key] for key in self.net.input]
        self.net.cfg['out_chans'] = sum(self.NUM_CHANS[key] for key in self.net.output)
