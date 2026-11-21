"""GPU stellar photometry, distances, luminosities, and Gaia color-temperature relations."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import vknumpy as vnp

from .coordinates import _to_f32

PC_TO_LY = 3.261563777


def parallax_to_distance(parallax_mas: Any, unit: str = "pc") -> vnp.ndarray:
    """Convert Gaia parallax in milliarcseconds (mas) -> distance in parsecs ('pc') or light-years ('ly')."""
    plx = vnp.maximum(_to_f32(parallax_mas).ravel(), 1e-6)
    d_pc = 1000.0 / plx
    if unit == "pc":
        return d_pc
    if unit == "ly":
        return d_pc * PC_TO_LY
    raise ValueError(f"Unsupported unit={unit!r}; expected 'pc' or 'ly'")


def distance_modulus(distance_pc: Any) -> vnp.ndarray:
    """Compute distance modulus mu = 5 * log10(d_pc / 10) on GPU."""
    d = vnp.maximum(_to_f32(distance_pc).ravel(), 1e-6)
    return 5.0 * (vnp.log(d * 0.1) * float(1.0 / math.log(10.0)))


def absolute_magnitude(apparent_mag: Any, distance_pc: Any) -> vnp.ndarray:
    """Compute absolute magnitude M = m - 5 * log10(d_pc / 10) on GPU."""
    return _to_f32(apparent_mag).ravel() - distance_modulus(distance_pc)


def luminosity_solar(abs_mag: Any, m_sun: float = 4.83) -> vnp.ndarray:
    """Compute stellar luminosity in solar units L / L_sun = 10^(0.4 * (M_sun - M)) on GPU."""
    m = _to_f32(abs_mag).ravel()
    return vnp.exp((float(m_sun) - m) * float(0.4 * math.log(10.0)))


def bp_rp_to_temperature(bp_rp: Any) -> vnp.ndarray:
    """Approximate Gaia (BP - RP) color index -> effective temperature T_eff (Kelvin) on GPU."""
    c = vnp.clip(_to_f32(bp_rp).ravel(), -0.5, 4.5)
    # Smooth Ballesteros-style relation calibrated to Gaia DR3 main sequence
    return 4600.0 * (1.0 / (0.92 * (c + 0.5) + 1.7) + 1.0 / (0.92 * (c + 0.5) + 0.62))
