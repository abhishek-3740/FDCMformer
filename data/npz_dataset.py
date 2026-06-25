import os
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, Any


class NPZ_Dataset(Dataset):
    """Loads SEN12MS-CR from either:
      - a single  <root>/<split>.npz  file, or
      - a folder  <root>/<split>/shard_XXXX.npz  (sharded layout)
    Each shard / file must contain keys: s1, s2, label, paths.
    """

    def __init__(self, root: str, split: str = 'all', data_range=1.0, crop_size=None):
        super().__init__()
        self.data_range = data_range
        self.crop_size = crop_size

        single = Path(root) / f'{split}.npz'
        shard_dir = Path(root) / split

        if single.exists():
            shards = [single]
        elif shard_dir.is_dir():
            shards = sorted(shard_dir.glob('shard_*.npz'))
            if not shards:
                raise FileNotFoundError(f'No shards found in {shard_dir}')
        else:
            raise FileNotFoundError(
                f'Neither {single} nor {shard_dir}/ exists.')

        self.sar_list, self.cloudy_list, self.target_list, self.paths_list = [], [], [], []
        for shard in shards:
            d = np.load(shard, allow_pickle=True)
            self.sar_list.append(d['s1'])
            self.cloudy_list.append(d['s2'])
            self.target_list.append(d['label'])
            self.paths_list.append(d['paths'])

        self.sar_list    = np.concatenate(self.sar_list,    axis=0)
        self.cloudy_list = np.concatenate(self.cloudy_list, axis=0)
        self.target_list = np.concatenate(self.target_list, axis=0)
        self.paths_list  = np.concatenate(self.paths_list,  axis=0)
        print(f'{split}: {len(self)} samples  '
              f's1={self.sar_list[0].shape}  s2={self.cloudy_list[0].shape}')

    def __len__(self) -> int:
        return len(self.sar_list)

    def __getitem__(self, idx) -> Dict[str, Any]:
        s   = {'SAR':    self.sar_list[idx].astype(np.float32)    * self.data_range,
               'cloudy': self.cloudy_list[idx].astype(np.float32) * self.data_range,
               'target': self.target_list[idx].astype(np.float32) * self.data_range}

        if self.crop_size is not None:
            h, w = s['SAR'].shape[-2:]
            cs = self.crop_size
            top  = np.random.randint(0, max(h - cs, 0) + 1)
            left = np.random.randint(0, max(w - cs, 0) + 1)
            for k in s:
                s[k] = s[k][..., top:top+cs, left:left+cs].copy()
        return s
