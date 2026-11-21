"""Astropy-style FlatLambdaCDM cosmology on Vulkan GPU."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import vknumpy as vnp

from .coordinates import _to_f32

C_KM_S = 299792.458


class FlatLambdaCDM:
    """Flat Lambda-CDM cosmology with vectorized GPU distance calculations."""

    def __init__(self, H0: float = 70.0, Om0: float = 0.3) -> None:
        self.H0 = float(H0)
        self.Om0 = float(Om0)
        self.Ode0 = 1.0 - self.Om0
        self.Dh = C_KM_S / self.H0  # Hubble distance in Mpc

    def E(self, z: Any) -> vnp.ndarray:
        zg = _to_f32(z).ravel()
        zp1 = 1.0 + zg
        return vnp.sqrt(self.Om0 * (zp1 * zp1 * zp1) + self.Ode0)

    def H(self, z: Any) -> vnp.ndarray:
        """Hubble parameter H(z) in km / s / Mpc on GPU."""
        return self.E(z) * self.H0

    def comoving_distance(self, z: Any, n_steps: int = 32) -> vnp.ndarray:
        """Line-of-sight comoving distance D_C(z) in Mpc via GPU Gauss-Legendre quadrature."""
        zg = _to_f32(z).ravel()
        nodes, weights = np.polynomial.legendre.leggauss(n_steps)
        # Map [-1, 1] -> [0, 1]
        u = 0.5 * (nodes + 1.0)
        w = 0.5 * weights
        acc = vnp.zeros(zg.shape, dtype=np.float32)
        for ui, wi in zip(u, w):
            zp1 = 1.0 + zg * float(ui)
            inv_e = 1.0 / vnp.sqrt(self.Om0 * (zp1 * zp1 * zp1) + self.Ode0)
            acc = acc + inv_e * float(wi)
        return acc * zg * self.Dh

    def luminosity_distance(self, z: Any) -> vnp.ndarray:
        """Luminosity distance D_L(z) = (1 + z) * D_C(z) in Mpc on GPU."""
        zg = _to_f32(z).ravel()
        return self.comoving_distance(zg) * (1.0 + zg)

    def angular_diameter_distance(self, z: Any) -> vnp.ndarray:
        """Angular diameter distance D_A(z) = D_C(z) / (1 + z) in Mpc on GPU."""
        zg = _to_f32(z).ravel()
        return self.comoving_distance(zg) / (1.0 + zg)

    def distmod(self, z: Any) -> vnp.ndarray:
        """Cosmological distance modulus mu(z) = 5 * log10(D_L / 10 pc) on GPU."""
        dl_mpc = vnp.maximum(self.luminosity_distance(z), 1e-9)
        return 5.0 * (vnp.log(dl_mpc) * float(1.0 / math.log(10.0))) + 25.0
