from typing import Any
import os
import glob

from .ecrformer_config import ECRformerConfig


def _find_dataset_roots(search_base='/kaggle/input'):
    """Auto-locate the two SEN12MS-CR parts regardless of mount prefix.

    Kaggle may mount datasets at /kaggle/input/<slug> OR at a deeper path like
    /kaggle/input/datasets/<user>/<slug>. We identify the parts by structure:
      * part1 = a directory that holds train/ val/ test/ subfolders with shards
      * part2 = a flat directory of shard_*.npz that is NOT part1 or its splits
    Returns (part1, part2_or_None); either may be None if not found.
    """
    if not os.path.isdir(search_base):
        return None, None

    part1 = None
    for root, dirs, _ in os.walk(search_base):
        if all(os.path.isdir(os.path.join(root, s)) for s in ('train', 'val', 'test')) \
                and glob.glob(os.path.join(root, 'train', 'shard_*.npz')):
            part1 = root
            break

    part1_splits = {os.path.join(part1, s) for s in ('train', 'val', 'test')} \
        if part1 else set()
    part2 = None
    for root, dirs, files in os.walk(search_base):
        if root == part1 or root in part1_splits:
            continue
        if any(f.startswith('shard_') and f.endswith('.npz') for f in files):
            part2 = root
            break

    return part1, part2


class ECRformerKaggleConfig(ECRformerConfig):
    """ECRformer config tuned for Kaggle's dual T4 (16 GB x2) environment.

    Dataset roots are auto-detected under /kaggle/input (any mount depth):
      part1 = the folder with train/ val/ test/ subfolders (train + val + test)
      part2 = the flat folder of extra shard_*.npz  (extra TRAIN shards)

    Run with 2 GPUs (recommended: launch as a subprocess from a notebook cell):
        !python train.py -c ecrformer_kaggle --devices 2 --strategy ddp
    In-kernel alternative (calling trainer.fit directly in the notebook):
        python train.py -c ecrformer_kaggle --devices 2 --strategy ddp_notebook
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        # ----- data: two Kaggle datasets merged for train only -----------
        part1, part2 = _find_dataset_roots()
        # Fall back to the conventional slugs if auto-detection finds nothing.
        self.dataset.root = part1 or '/kaggle/input/sen12ms-cr-spring-part1'
        self.dataset.train_extra_roots = (
            [part2] if part2 else ['/kaggle/input/sen12ms-cr-spring-part2'])
        print(f'[ecrformer_kaggle] part1 (train/val/test): {self.dataset.root}')
        print(f'[ecrformer_kaggle] part2 (extra train)   : {self.dataset.train_extra_roots[0]}')
        # part1 has train/, val/, test/ subfolders; part2 is a flat shard dir.
        self.dataset.split = ['train', 'val', 'test']

        # ----- multi-GPU / precision -------------------------------------
        # T4 has no bf16; use fp16 mixed precision for speed + headroom.
        self.optim.accelerator = 'gpu'
        self.optim.precision = '16-mixed'
        self.train.strategy = 'ddp'             # subprocess launch (`!python train.py`)
        self.train.gpu = 2                      # both T4s

        # Per-GPU batch. Effective batch = train_bs * accumulate * num_gpus.
        # 4 * 2 * 2 = 16 (matches the single-GPU 4*4 default effective batch).
        self.train.train_bs = 4
        self.train.valid_bs = 8
        self.optim.accumulate_grad_batches = 2

        # Kaggle gives ~4 CPU cores; keep workers modest (per DDP process).
        self.train.num_workers = 2

        # ----- training schedule (professional ~80-epoch run) ------------
        self.train.max_epoch = 80
        self.train.early_stop = 15            # patience on valid_loss
        # MultiStepLR rescaled for an 80-epoch budget (halves LR at each step).
        self.train.lr_milestones = [40, 60, 70, 75]
        self.train.lr_gamma = 0.5
        self.optim.log_every_n_steps = 20

        # ----- checkpointing / graceful stop within Kaggle's 9h cap ------
        self.train.save_top_k = 3             # keep the 3 best models
        self.train.ckpt_every_n_min = 30      # refresh last.ckpt every 30 min
        # Stop cleanly (and checkpoint) ~30 min before Kaggle's hard 9h kill.
        self.train.max_time = '00:08:30:00'   # DD:HH:MM:SS

        # Persist checkpoints/logs to the notebook's writable output dir.
        self.train.save_dir = '/kaggle/working/experiments'
