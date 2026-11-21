"""Astropy-style GPU sky coordinates (ICRS <-> Galactic, 3D Cartesian, separation, Aitoff)."""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import numpy as np
import vknumpy as vnp

# Exact Hipparcos / IAU J2000 ICRS -> Galactic 3x3 orthogonal rotation matrix
_ICRS_TO_GAL = np.array([
    [-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
    [ 0.4941094278755837, -0.4448296299600112,  0.7469822444972189],
    [-0.8676661490190047, -0.1980763734312015,  0.4559837761750669],
], dtype=np.float32)

_GAL_TO_ICRS = _ICRS_TO_GAL.T.copy()


def _to_f32(a: Any) -> vnp.ndarray:
    if hasattr(a, "values") and isinstance(a.values, vnp.ndarray):
        arr = a.values
        return arr if arr.dtype == np.float32 else arr.astype(np.float32)
    if isinstance(a, vnp.ndarray):
        return a if a.dtype == np.float32 else a.astype(np.float32)
    return vnp.asarray(a, dtype=np.float32)


def spherical_to_cartesian(
    lon_deg: Any,
    lat_deg: Any,
    distance: Optional[Any] = None,
) -> Tuple[vnp.ndarray, vnp.ndarray, vnp.ndarray]:
    """Convert spherical (lon, lat) in degrees + optional distance -> 3D (x, y, z) on GPU."""
    d2r = float(math.pi / 180.0)
    lon = _to_f32(lon_deg).ravel() * d2r
    lat = _to_f32(lat_deg).ravel() * d2r
    clat = vnp.cos(lat)
    x = clat * vnp.cos(lon)
    y = clat * vnp.sin(lon)
    z = vnp.sin(lat)
    if distance is not None:
        r = _to_f32(distance).ravel()
        x = x * r
        y = y * r
        z = z * r
    return x, y, z


def _rotate_sky(lon_deg: Any, lat_deg: Any, mat: np.ndarray) -> Tuple[vnp.ndarray, vnp.ndarray]:
    x, y, z = spherical_to_cartesian(lon_deg, lat_deg)
    # Apply 3x3 rotation directly on GPU via elementwise ops (zero stack overhead)
    m = mat
    rx = x * float(m[0, 0]) + y * float(m[0, 1]) + z * float(m[0, 2])
    ry = x * float(m[1, 0]) + y * float(m[1, 1]) + z * float(m[1, 2])
    rz = x * float(m[2, 0]) + y * float(m[2, 1]) + z * float(m[2, 2])
    rz = vnp.clip(rz, -1.0, 1.0)
    r2d = float(180.0 / math.pi)
    lat_out = vnp.arcsin(rz) * r2d
    if hasattr(vnp, "arctan2"):
        lon_out = (vnp.arctan2(ry, rx) * r2d + 360.0) % 360.0
    else:
        lon_np = (np.degrees(np.arctan2(ry.to_numpy(), rx.to_numpy())).astype(np.float32) + 360.0) % 360.0
        lon_out = vnp.array(lon_np, dtype=np.float32)
    return lon_out, lat_out


def icrs_to_galactic(ra_deg: Any, dec_deg: Any) -> Tuple[vnp.ndarray, vnp.ndarray]:
    """Transform ICRS J2000 (RA, Dec) degrees -> Galactic (l, b) degrees on Vulkan GPU."""
    return _rotate_sky(ra_deg, dec_deg, _ICRS_TO_GAL)


def galactic_to_icrs(l_deg: Any, b_deg: Any) -> Tuple[vnp.ndarray, vnp.ndarray]:
    """Transform Galactic (l, b) degrees -> ICRS J2000 (RA, Dec) degrees on Vulkan GPU."""
    return _rotate_sky(l_deg, b_deg, _GAL_TO_ICRS)


def angular_separation(
    ra1_deg: Any,
    dec1_deg: Any,
    ra2_deg: Any,
    dec2_deg: Any,
    unit: str = "deg",
) -> vnp.ndarray:
    """Great-circle spherical angular separation on Vulkan GPU ('deg', 'arcmin', 'arcsec', 'rad')."""
    d2r = float(math.pi / 180.0)
    p1 = _to_f32(dec1_deg).ravel() * d2r
    p2 = _to_f32(dec2_deg).ravel() * d2r
    dp = p2 - p1
    dl = (_to_f32(ra2_deg).ravel() - _to_f32(ra1_deg).ravel()) * d2r
    s_dp = vnp.sin(dp * 0.5)
    s_dl = vnp.sin(dl * 0.5)
    a = vnp.clip(s_dp * s_dp + vnp.cos(p1) * vnp.cos(p2) * (s_dl * s_dl), 0.0, 1.0)
    rad = 2.0 * vnp.arcsin(vnp.sqrt(a))
    if unit == "rad":
        return rad
    deg = rad * float(180.0 / math.pi)
    if unit == "deg":
        return deg
    if unit == "arcmin":
        return deg * 60.0
    if unit == "arcsec":
        return deg * 3600.0
    raise ValueError(f"Unsupported unit={unit!r}; expected 'deg', 'arcmin', 'arcsec', or 'rad'")


def aitoff_projection(lon_deg: Any, lat_deg: Any) -> Tuple[vnp.ndarray, vnp.ndarray]:
    """All-sky Hammer-Aitoff equal-area projection on GPU (lon in [-180, 180] or [0, 360], lat in [-90, 90])."""
    lon_raw = _to_f32(lon_deg).ravel()
    # Wrap [0, 360] -> [-180, 180] so Galactic Center l=0 sits cleanly at the center of the map
    lon_w = ((lon_raw + 180.0) % 360.0) - 180.0
    # Astronomical convention: longitude increases to the left (East-left)
    d2r = float(math.pi / 180.0)
    lam = -lon_w * d2r
    phi = _to_f32(lat_deg).ravel() * d2r

    cphi = vnp.cos(phi)
    half_lam = lam * 0.5
    denom = vnp.sqrt(vnp.maximum(1.0 + cphi * vnp.cos(half_lam), 1e-6))
    x = (2.0 * math.sqrt(2.0) * cphi * vnp.sin(half_lam)) / denom
    y = (math.sqrt(2.0) * vnp.sin(phi)) / denom
    return x, y


class SkyCoord:
    """Astropy-style GPU SkyCoord supporting ICRS <-> Galactic transforms, 3D Cartesian, and cone search."""

    def __init__(
        self,
        ra: Any,
        dec: Any,
        distance: Optional[Any] = None,
        frame: str = "icrs",
    ) -> None:
        self.frame = frame.lower()
        self.ra = _to_f32(ra).ravel()
        self.dec = _to_f32(dec).ravel()
        self.distance = _to_f32(distance).ravel() if distance is not None else None

    @property
    def l(self) -> vnp.ndarray:
        return self.galactic.ra

    @property
    def b(self) -> vnp.ndarray:
        return self.galactic.dec

    @property
    def galactic(self) -> "SkyCoord":
        if self.frame == "galactic":
            return self
        l_g, b_g = icrs_to_galactic(self.ra, self.dec)
        return SkyCoord(l_g, b_g, distance=self.distance, frame="galactic")

    @property
    def icrs(self) -> "SkyCoord":
        if self.frame == "icrs":
            return self
        ra_g, dec_g = galactic_to_icrs(self.ra, self.dec)
        return SkyCoord(ra_g, dec_g, distance=self.distance, frame="icrs")

    def cartesian(self) -> Tuple[vnp.ndarray, vnp.ndarray, vnp.ndarray]:
        return spherical_to_cartesian(self.ra, self.dec, self.distance)

    def separation(self, other: "SkyCoord", unit: str = "deg") -> vnp.ndarray:
        o = other.icrs if self.frame == "icrs" else other.galactic
        return angular_separation(self.ra, self.dec, o.ra, o.dec, unit=unit)

    def cone_search(self, ra0: float, dec0: float, radius_deg: float) -> vnp.ndarray:
        """Return boolean GPU mask of sources within `radius_deg` of `(ra0, dec0)`."""
        n = self.ra.size
        r0 = vnp.full((n,), float(ra0), dtype=np.float32)
        d0 = vnp.full((n,), float(dec0), dtype=np.float32)
        sep = angular_separation(self.ra, self.dec, r0, d0, unit="deg")
        return sep <= float(radius_deg)

    def __len__(self) -> int:
        return int(self.ra.size)

    def __repr__(self) -> str:
        return f"SkyCoord(n={len(self):,}, frame={self.frame!r}, has_distance={self.distance is not None})"
