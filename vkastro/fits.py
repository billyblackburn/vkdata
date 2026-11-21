"""Lightweight zero-dependency FITS image reader directly into vknumpy GPU arrays."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
import vknumpy as vnp

_BITPIX_DTYPE = {
    8: ">u1",
    16: ">i2",
    32: ">i4",
    -32: ">f4",
    -64: ">f8",
}


def read_fits(path: Union[str, Path]) -> Tuple[vnp.ndarray, Dict[str, Any]]:
    """Read the primary HDU image from a FITS file directly into a float32 vknumpy GPU array."""
    raw = Path(path).read_bytes()
    header: Dict[str, Any] = {}
    pos = 0
    done = False
    while pos < len(raw) and not done:
        block = raw[pos : pos + 2880].decode("ascii", errors="ignore")
        pos += 2880
        for i in range(0, 2880, 80):
            card = block[i : i + 80]
            key = card[:8].strip()
            if key == "END":
                done = True
                break
            if card[8:10] == "= ":
                val_str = card[10:].split("/")[0].strip()
                if val_str.startswith("'") and val_str.endswith("'"):
                    header[key] = val_str[1:-1].strip()
                elif val_str in ("T", "F"):
                    header[key] = (val_str == "T")
                else:
                    try:
                        header[key] = int(val_str)
                    except ValueError:
                        try:
                            header[key] = float(val_str)
                        except ValueError:
                            header[key] = val_str

    naxis = int(header.get("NAXIS", 0))
    if naxis == 0:
        return vnp.zeros((0,), dtype=np.float32), header

    bitpix = int(header["BITPIX"])
    if bitpix not in _BITPIX_DTYPE:
        raise ValueError(f"Unsupported FITS BITPIX={bitpix}")
    dims = [int(header[f"NAXIS{k}"]) for k in range(1, naxis + 1)]
    shape = tuple(reversed(dims))
    count = int(np.prod(shape))
    dt = np.dtype(_BITPIX_DTYPE[bitpix])
    arr_np = np.frombuffer(raw, dtype=dt, count=count, offset=pos).reshape(shape).astype(np.float32)

    bscale = float(header.get("BSCALE", 1.0))
    bzero = float(header.get("BZERO", 0.0))
    if bscale != 1.0 or bzero != 0.0:
        arr_np = arr_np * bscale + bzero
    return vnp.array(arr_np, dtype=np.float32), header
