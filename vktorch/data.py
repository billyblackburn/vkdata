"""vktorch.data: TensorDataset and shuffled GPU mini-batch DataLoader."""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as _np
import vknumpy as vnp

from .tensor import Tensor


class Dataset:
    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class TensorDataset(Dataset):
    """Wraps aligned feature/target arrays for mini-batch iteration."""

    def __init__(self, *arrays) -> None:
        if not arrays:
            raise ValueError("TensorDataset requires at least one array")
        self.arrays = [
            a.to_numpy() if isinstance(a, (Tensor, vnp.ndarray)) else _np.asarray(a)
            for a in arrays
        ]
        n = len(self.arrays[0])
        for a in self.arrays[1:]:
            if len(a) != n:
                raise ValueError("All arrays in TensorDataset must have the same length")
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx):
        return tuple(a[idx] for a in self.arrays)


class DataLoader:
    """Mini-batch iterator yielding ``vktorch.Tensor`` (float) / ``ndarray`` (int) batches."""

    def __init__(
        self,
        dataset: TensorDataset,
        batch_size: int = 64,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self._rng = _np.random.default_rng(seed) if seed is not None else _np.random.default_rng()

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Tuple]:
        n = len(self.dataset)
        order = self._rng.permutation(n) if self.shuffle else _np.arange(n)
        for start in range(0, n, self.batch_size):
            end = start + self.batch_size
            if end > n and self.drop_last:
                break
            idx = order[start:end]
            batch = []
            for arr in self.dataset.arrays:
                sub = _np.ascontiguousarray(arr[idx])
                if _np.issubdtype(sub.dtype, _np.floating):
                    batch.append(Tensor(sub.astype(_np.float32), requires_grad=False))
                else:
                    batch.append(sub)
            yield tuple(batch)
