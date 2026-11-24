"""vktorch.optim: VRAM-resident Vulkan GPU optimizers (SGD, Adam, AdamW) & LR schedulers."""

from __future__ import annotations

import math
from typing import Iterable, List, Optional

import numpy as _np
import vknumpy as vnp

from .tensor import Tensor


class Optimizer:
    """Base optimizer class operating directly on Vulkan GPU ``Tensor`` parameters."""

    def __init__(self, params: Iterable[Tensor], lr: float) -> None:
        self.params: List[Tensor] = [p for p in params if p.requires_grad]
        self.lr = float(lr)

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent with momentum & decoupled weight decay on Vulkan GPU."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-2,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr=lr)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self._buf: List[Optional[vnp.ndarray]] = [None] * len(self.params)

    def step(self) -> None:
        lr = self.lr
        mu = self.momentum
        wd = self.weight_decay
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if wd != 0.0:
                g = g + p.data * wd
            if mu != 0.0:
                b = self._buf[i]
                b = g if b is None else (b * mu + g)
                self._buf[i] = b
                g = b
            p.data = p.data - g * lr


class Adam(Optimizer):
    """Vulkan GPU Adam optimizer."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr=lr)
        self.beta1 = float(betas[0])
        self.beta2 = float(betas[1])
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self._m: List[vnp.ndarray] = [vnp.zeros(p.shape, dtype=_np.float32) for p in self.params]
        self._v: List[vnp.ndarray] = [vnp.zeros(p.shape, dtype=_np.float32) for p in self.params]
        self.t: int = 0

    def step(self) -> None:
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        bc1 = 1.0 - (b1 ** self.t)
        bc2 = 1.0 - (b2 ** self.t)
        step_size = self.lr / bc1
        inv_sqrt_bc2 = 1.0 / math.sqrt(bc2)

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay != 0.0:
                g = g + p.data * self.weight_decay
            self._m[i] = self._m[i] * b1 + g * (1.0 - b1)
            self._v[i] = self._v[i] * b2 + (g * g) * (1.0 - b2)
            denom = vnp.sqrt(self._v[i]) * inv_sqrt_bc2 + self.eps
            p.data = p.data - (self._m[i] / denom) * step_size


class AdamW(Adam):
    """Vulkan GPU AdamW optimizer (decoupled weight decay)."""

    def step(self) -> None:
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        bc1 = 1.0 - (b1 ** self.t)
        bc2 = 1.0 - (b2 ** self.t)
        step_size = self.lr / bc1
        inv_sqrt_bc2 = 1.0 / math.sqrt(bc2)
        wd_scale = 1.0 - self.lr * self.weight_decay

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay != 0.0:
                p.data = p.data * wd_scale
            self._m[i] = self._m[i] * b1 + g * (1.0 - b1)
            self._v[i] = self._v[i] * b2 + (g * g) * (1.0 - b2)
            denom = vnp.sqrt(self._v[i]) * inv_sqrt_bc2 + self.eps
            p.data = p.data - (self._m[i] / denom) * step_size


def clip_grad_norm_(params: Iterable[Tensor], max_norm: float) -> float:
    """Clip global L2 gradient norm in-place on GPU."""
    ps = [p for p in params if p.grad is not None]
    if not ps:
        return 0.0
    total_sq = sum(float(vnp.sum(p.grad * p.grad).item()) for p in ps)
    total_norm = math.sqrt(max(total_sq, 0.0))
    if total_norm > max_norm > 0.0:
        scale = float(max_norm) / (total_norm + 1e-6)
        for p in ps:
            p.grad = p.grad * scale
    return total_norm


class StepLR:
    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.5) -> None:
        self.optimizer = optimizer
        self.step_size = max(1, int(step_size))
        self.gamma = float(gamma)
        self.base_lr = optimizer.lr
        self.epoch = 0

    def step(self) -> None:
        self.epoch += 1
        self.optimizer.lr = self.base_lr * (self.gamma ** (self.epoch // self.step_size))


class CosineAnnealingLR:
    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0.0) -> None:
        self.optimizer = optimizer
        self.T_max = max(1, int(T_max))
        self.eta_min = float(eta_min)
        self.base_lr = optimizer.lr
        self.epoch = 0

    def step(self) -> None:
        self.epoch += 1
        t = min(self.epoch, self.T_max)
        self.optimizer.lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1.0 + math.cos(math.pi * t / self.T_max))
