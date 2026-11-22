"""Zero-copy Vulkan GPU 6-level 3D Barnes-Hut Morton Octree N-body solver."""

from __future__ import annotations

import numpy as _np
import vknumpy as vnp

from ._backend import barnes_hut_step_gpu

NUM_LEAVES = 32768      # 32^3 Morton leaf cells
NUM_TREE_NODES = 37449  # 1 + 8 + 64 + 512 + 4096 + 32768


class BarnesHut3D:
    """6-level 3D Barnes-Hut Morton octree N-body + elastic collision solver on Vulkan GPU.

    Parameters
    ----------
    pos_mass : array-like of shape (N, 4), float32
        Initial ``[x, y, z, mass]`` per body.
    vel_gal : array-like of shape (N, 4), float32
        Initial ``[vx, vy, vz, gal_id]`` per body. The integer part of ``w`` encodes
        galaxy/group ID; the fractional part ``[0, 0.99)`` stores the decaying
        elastic-collision flash intensity.
    G : float
        Gravitational constant.
    softening_sq : float
        Plummer softening squared ``eps^2``.
    dt : float
        Integration timestep per substep.
    damping : float
        Per-step velocity damping factor (e.g. ``0.9997``).
    theta : float
        Barnes-Hut opening angle criterion ``s / d < theta`` (smaller = more accurate).
    restitution : float
        Coefficient of restitution for near-field elastic sphere-sphere collisions.
    world_half : float
        Half-width of the cubic octree domain ``[-world_half, +world_half]^3``.
    """

    def __init__(
        self,
        pos_mass,
        vel_gal,
        *,
        G: float = 0.085,
        softening_sq: float = 0.20,
        dt: float = 0.014,
        damping: float = 0.9997,
        theta: float = 0.88,
        restitution: float = 0.65,
        world_half: float = 48.0,
    ) -> None:
        pm = _np.ascontiguousarray(pos_mass, dtype=_np.float32)
        vg = _np.ascontiguousarray(vel_gal, dtype=_np.float32)
        if pm.ndim != 2 or pm.shape[1] != 4:
            raise ValueError(f"pos_mass must have shape (N, 4), got {pm.shape}")
        if vg.shape != pm.shape:
            raise ValueError(f"vel_gal must match pos_mass shape {pm.shape}, got {vg.shape}")

        self.n = int(pm.shape[0])
        self.G = float(G)
        self.softening_sq = float(softening_sq)
        self.dt = float(dt)
        self.damping = float(damping)
        self.theta = float(theta)
        self.restitution = float(restitution)
        self.world_half = float(world_half)

        # Ping-pong state buffers on GPU
        self._pm_a = vnp.array(pm, dtype=_np.float32)
        self._pm_b = vnp.ndarray._empty_uninit((self.n, 4), _np.float32)
        self._vg_a = vnp.array(vg, dtype=_np.float32)
        self._vg_b = vnp.ndarray._empty_uninit((self.n, 4), _np.float32)

        # Octree scratch buffers on GPU (reused every step, zero allocations)
        self._leaf_heads = vnp.ndarray._empty_uninit((NUM_LEAVES,), _np.int32)
        self._particle_next = vnp.ndarray._empty_uninit((self.n,), _np.int32)
        self._tree_com = vnp.ndarray._empty_uninit((NUM_TREE_NODES, 4), _np.float32)
        self._tree_size = vnp.ndarray._empty_uninit((NUM_TREE_NODES,), _np.float32)

        self._parity = 0
        self.steps = 0

    @property
    def pos_mass(self) -> vnp.ndarray:
        """Current GPU ``(N, 4)`` float32 ``[x, y, z, mass]`` buffer."""
        return self._pm_a if self._parity == 0 else self._pm_b

    @property
    def vel_gal(self) -> vnp.ndarray:
        """Current GPU ``(N, 4)`` float32 ``[vx, vy, vz, gal_id + flash]`` buffer."""
        return self._vg_a if self._parity == 0 else self._vg_b

    def reset(self, pos_mass, vel_gal) -> None:
        """Reset particle state on GPU without reallocating octree buffers."""
        pm = _np.ascontiguousarray(pos_mass, dtype=_np.float32)
        vg = _np.ascontiguousarray(vel_gal, dtype=_np.float32)
        if pm.shape != (self.n, 4) or vg.shape != (self.n, 4):
            raise ValueError(f"Expected shape ({self.n}, 4)")
        self._pm_a = vnp.array(pm, dtype=_np.float32)
        self._vg_a = vnp.array(vg, dtype=_np.float32)
        self._parity = 0
        self.steps = 0

    def step(self, n_substeps: int = 1) -> None:
        """Advance the N-body system by ``n_substeps`` Barnes-Hut octree steps on GPU."""
        for _ in range(int(n_substeps)):
            if self._parity == 0:
                src_pm, src_vg = self._pm_a, self._vg_a
                dst_pm, dst_vg = self._pm_b, self._vg_b
            else:
                src_pm, src_vg = self._pm_b, self._vg_b
                dst_pm, dst_vg = self._pm_a, self._vg_a

            barnes_hut_step_gpu(
                src_pm, src_vg,
                dst_pm, dst_vg,
                self._leaf_heads,
                self._particle_next,
                self._tree_com,
                self._tree_size,
                self.n,
                self.G,
                self.softening_sq,
                self.dt,
                self.damping,
                self.theta,
                self.restitution,
                self.world_half,
            )
            self._parity ^= 1
            self.steps += 1
