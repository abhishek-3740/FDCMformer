import ast
import struct
import zipfile
import bisect
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Any, List, Sequence, Union


# ---------------------------------------------------------------------------
# Low-level helpers: memory-map arrays inside an *uncompressed* .npz file.
#
# An uncompressed .npz is just a ZIP archive of .npy entries stored with no
# compression (np.savez, NOT np.savez_compressed). That means each array's raw
# bytes sit contiguously inside the .npz on disk, so we can np.memmap straight
# into the parent file at the right offset and never load the whole shard into
# RAM. Random access only faults in the pages it actually touches (and the OS
# page cache is shared across workers), which is exactly what we need to train
# on tens of GB of shards inside Kaggle's ~30 GB RAM.
# ---------------------------------------------------------------------------


def _read_npy_header(f):
    """Parse a .npy header from a file-like object positioned at its start.

    Returns (dtype, fortran_order, shape) and leaves the cursor at the first
    data byte. Self-contained (no private numpy APIs) for version robustness.
    """
    magic = f.read(6)
    if magic != b'\x93NUMPY':
        raise ValueError('Not a .npy stream (bad magic).')
    major = f.read(1)[0]
    f.read(1)  # minor
    if major == 1:
        (hlen,) = struct.unpack('<H', f.read(2))
    else:
        (hlen,) = struct.unpack('<I', f.read(4))
    header = f.read(hlen).decode('latin1')
    d = ast.literal_eval(header)
    return np.dtype(d['descr']), bool(d['fortran_order']), tuple(d['shape'])


def _zip_entry_data_offset(zf: zipfile.ZipFile, zinfo: zipfile.ZipInfo) -> int:
    """Absolute byte offset of an entry's *raw data* within the .npz file.

    Must read the local file header (its extra-field length can differ from
    the central directory's), then skip past it.
    """
    fp = zf.fp
    fp.seek(zinfo.header_offset)
    local = fp.read(30)
    if local[:4] != b'PK\x03\x04':
        raise ValueError('Bad ZIP local file header.')
    fname_len = struct.unpack('<H', local[26:28])[0]
    extra_len = struct.unpack('<H', local[28:30])[0]
    return zinfo.header_offset + 30 + fname_len + extra_len


class _ShardMeta:
    """Per-shard metadata (lazy): array offsets/dtypes/shapes, no data loaded."""

    DATA_KEYS = ('s1', 's2', 'label')

    def __init__(self, path: Path):
        self.path = Path(path)
        self.compressed = False
        self.specs: Dict[str, tuple] = {}   # key -> (dtype, shape, data_offset)
        self.num_samples = 0
        self._scan()

    def _scan(self):
        with zipfile.ZipFile(self.path, 'r') as zf:
            entries = {zi.filename: zi for zi in zf.infolist()}
            lengths = []
            for key in self.DATA_KEYS:
                name = f'{key}.npy'
                if name not in entries:
                    raise KeyError(f'{self.path} is missing array "{key}".')
                zinfo = entries[name]
                if zinfo.compress_type != zipfile.ZIP_STORED:
                    # Compressed shard: cannot memmap; fall back to np.load later.
                    self.compressed = True
                    return
                base = _zip_entry_data_offset(zf, zinfo)
                with zf.open(zinfo) as f:
                    dtype, fortran, shape = _read_npy_header(f)
                    header_len = f.tell()
                if fortran:
                    raise ValueError(f'Fortran-order arrays unsupported: {name}')
                self.specs[key] = (dtype, shape, base + header_len)
                lengths.append(shape[0])
            if len(set(lengths)) != 1:
                raise ValueError(f'Mismatched sample counts in {self.path}: {lengths}')
            self.num_samples = lengths[0]

        if self.compressed:
            # Determine length from a lightweight load of one array's shape.
            with np.load(self.path, allow_pickle=True) as d:
                self.num_samples = d['s1'].shape[0]


class NPZ_Dataset(Dataset):
    """SEN12MS-CR loader with O(1) RAM via lazy memory-mapping.

    Accepts, for a given ``split``, one or more ``root`` directories. For each
    root it looks for (in order):
      1. ``<root>/<split>.npz``               (single packed file)
      2. ``<root>/<split>/shard_*.npz``       (sharded layout, e.g. part1)
      3. ``<root>/shard_*.npz``               (flat shard dir, e.g. Kaggle part2)

    Passing a list of roots merges all discovered shards (deduplicated). This is
    how the SEN12MS-CR "spring" set split across two Kaggle datasets is joined:
    train reads ``part1/train`` + ``part2`` while val/test read only ``part1``.

    Each shard must contain keys: s1, s2, label, (paths). Uncompressed shards
    (np.savez) are memory-mapped; compressed ones fall back to a cached np.load.
    """

    def __init__(self,
                 root: Union[str, Sequence[str]],
                 split: str = 'all',
                 data_range: float = 1.0,
                 crop_size: int = None):
        super().__init__()
        self.data_range = data_range
        self.crop_size = crop_size

        shards = self._resolve_shards(root, split)
        if not shards:
            roots_str = root if isinstance(root, (str, Path)) else list(root)
            raise FileNotFoundError(
                f'No shards found for split="{split}" under roots={roots_str!r}.')

        # Lightweight metadata scan (reads only headers, not array data).
        self.shards: List[_ShardMeta] = [_ShardMeta(s) for s in shards]
        self.shard_lengths = [m.num_samples for m in self.shards]
        self.cum_lengths = np.cumsum([0] + self.shard_lengths).tolist()
        self.total = self.cum_lengths[-1]

        # Per-shard array handles, created lazily per-process/worker (fork/spawn
        # safe: nothing heavy is pickled, memmaps are opened on first touch).
        self._mm_cache: Dict[int, Dict[str, np.ndarray]] = {}

        # Paths are tiny (unicode arrays); load eagerly for test.py compatibility.
        self.paths_list = self._load_paths(shards)

        ref = self.shards[0].specs.get('s1')
        ref_shape = ref[1][1:] if ref is not None else '?'
        print(f'{split}: {self.total} samples across {len(self.shards)} shard(s) '
              f'(memmap)  s1[0]={ref_shape}')

    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve_shards(root, split) -> List[Path]:
        roots = [root] if isinstance(root, (str, Path)) else list(root)
        found: List[Path] = []
        seen = set()
        for r in roots:
            r = Path(r)
            single = r / f'{split}.npz'
            sub = r / split
            candidates: List[Path] = []
            if single.exists():
                candidates = [single]
            elif sub.is_dir() and list(sub.glob('shard_*.npz')):
                candidates = sorted(sub.glob('shard_*.npz'))
            elif r.is_dir() and list(r.glob('shard_*.npz')):
                # Flat directory of shards (Kaggle "part2" with no subfolders).
                candidates = sorted(r.glob('shard_*.npz'))
            for c in candidates:
                key = c.resolve()
                if key not in seen:
                    seen.add(key)
                    found.append(c)
        return found

    @staticmethod
    def _load_paths(shards) -> np.ndarray:
        parts = []
        for s in shards:
            try:
                with np.load(s, allow_pickle=True) as d:
                    if 'paths' in d.files:
                        parts.append(np.asarray(d['paths']))
            except Exception:
                pass
        if parts:
            return np.concatenate(parts, axis=0)
        return np.empty((0,), dtype=object)

    def _get_shard_arrays(self, shard_idx: int) -> Dict[str, np.ndarray]:
        arrays = self._mm_cache.get(shard_idx)
        if arrays is not None:
            return arrays
        meta = self.shards[shard_idx]
        if meta.compressed:
            with np.load(meta.path, allow_pickle=True) as d:
                arrays = {k: np.asarray(d[k]) for k in _ShardMeta.DATA_KEYS}
        else:
            arrays = {}
            for key, (dtype, shape, offset) in meta.specs.items():
                arrays[key] = np.memmap(meta.path, mode='r', dtype=dtype,
                                        shape=shape, offset=offset)
        self._mm_cache[shard_idx] = arrays
        return arrays

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx) -> Dict[str, Any]:
        if idx < 0:
            idx += self.total
        shard_idx = bisect.bisect_right(self.cum_lengths, idx) - 1
        local = idx - self.cum_lengths[shard_idx]
        arrays = self._get_shard_arrays(shard_idx)

        # np.array(..) forces a copy of just this sample out of the memmap.
        s = {
            'SAR':    np.array(arrays['s1'][local], dtype=np.float32) * self.data_range,
            'cloudy': np.array(arrays['s2'][local], dtype=np.float32) * self.data_range,
            'target': np.array(arrays['label'][local], dtype=np.float32) * self.data_range,
        }

        if self.crop_size is not None:
            h, w = s['SAR'].shape[-2:]
            cs = self.crop_size
            top = np.random.randint(0, max(h - cs, 0) + 1)
            left = np.random.randint(0, max(w - cs, 0) + 1)
            for k in s:
                s[k] = s[k][..., top:top + cs, left:left + cs].copy()
        return s
