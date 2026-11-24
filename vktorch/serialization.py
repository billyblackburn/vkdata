"""vktorch.serialization: Save and load model state dicts as compact .npz checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import numpy as _np


def save(state_dict: Dict[str, _np.ndarray], path: Union[str, Path]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    _np.savez(str(p), **{k: _np.asarray(v, dtype=_np.float32) for k, v in state_dict.items()})


def load(path: Union[str, Path]) -> Dict[str, _np.ndarray]:
    with _np.load(str(path)) as data:
        return {k: data[k].astype(_np.float32) for k in data.files}
