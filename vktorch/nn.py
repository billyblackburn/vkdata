"""vktorch.nn: PyTorch-compatible neural network layers, containers, and losses on Vulkan GPU."""

from __future__ import annotations

import math
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as _np
import vknumpy as vnp

from . import functional as F
from .tensor import Tensor, no_grad


class Parameter(Tensor):
    """Trainable parameter Tensor (automatically registered on ``Module`` attributes)."""

    def __init__(self, data, *, requires_grad: bool = True) -> None:
        super().__init__(data, requires_grad=requires_grad, dtype=_np.float32)


class Module:
    """Base class for all ``vktorch`` neural network modules (mirrors ``torch.nn.Module``)."""

    def __init__(self) -> None:
        self.training: bool = True

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _iter_children(self) -> Iterator[Tuple[str, object]]:
        for name, val in self.__dict__.items():
            if name.startswith("_"):
                continue
            yield name, val

    def named_parameters(self, prefix: str = "") -> Iterator[Tuple[str, Parameter]]:
        seen = set()

        def _walk(obj, pref: str):
            if isinstance(obj, Parameter):
                if id(obj) not in seen:
                    seen.add(id(obj))
                    yield pref.rstrip("."), obj
            elif isinstance(obj, Module):
                for name, val in obj._iter_children():
                    sub_pref = f"{pref}{name}." if not isinstance(val, Parameter) else f"{pref}{name}"
                    yield from _walk(val, sub_pref)
            elif isinstance(obj, (list, tuple)):
                for i, item in enumerate(obj):
                    yield from _walk(item, f"{pref}{i}.")
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    yield from _walk(v, f"{pref}{k}.")

        yield from _walk(self, prefix)

    def parameters(self) -> List[Parameter]:
        return [p for _, p in self.named_parameters()]

    def num_parameters(self) -> int:
        return sum(p.size for p in self.parameters())

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = None

    def train(self, mode: bool = True) -> "Module":
        self.training = bool(mode)
        for _, val in self._iter_children():
            if isinstance(val, Module):
                val.train(mode)
            elif isinstance(val, (list, tuple)):
                for item in val:
                    if isinstance(item, Module):
                        item.train(mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)

    def state_dict(self) -> Dict[str, _np.ndarray]:
        return {k: p.to_numpy().copy() for k, p in self.named_parameters()}

    def load_state_dict(self, state: Dict[str, _np.ndarray], strict: bool = True) -> None:
        named = dict(self.named_parameters())
        missing = [k for k in named if k not in state]
        if strict and missing:
            raise KeyError(f"Missing keys in state_dict: {missing}")
        for k, p in named.items():
            if k in state:
                arr = _np.asarray(state[k], dtype=_np.float32)
                if arr.shape != p.shape:
                    raise ValueError(f"Shape mismatch for '{k}': expected {p.shape}, got {arr.shape}")
                p.data = vnp.array(arr, dtype=_np.float32)


class ModuleList(Module):
    def __init__(self, modules: Optional[List[Module]] = None) -> None:
        super().__init__()
        self.modules: List[Module] = list(modules) if modules is not None else []

    def append(self, m: Module) -> "ModuleList":
        self.modules.append(m)
        return self

    def __iter__(self):
        return iter(self.modules)

    def __len__(self) -> int:
        return len(self.modules)

    def __getitem__(self, idx: int) -> Module:
        return self.modules[idx]


class Sequential(Module):
    def __init__(self, *layers: Module) -> None:
        super().__init__()
        self.layers: List[Module] = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class Linear(Module):
    """Fully-connected affine layer ``y = x @ W^T + b`` on Vulkan GPU."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, seed: Optional[int] = None) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        rng = _np.random.default_rng(seed) if seed is not None else _np.random.default_rng()
        bound = 1.0 / math.sqrt(float(in_features))
        w = rng.uniform(-bound, bound, size=(out_features, in_features)).astype(_np.float32)
        self.weight = Parameter(w)
        if bias:
            b = rng.uniform(-bound, bound, size=(out_features,)).astype(_np.float32)
            self.bias: Optional[Parameter] = Parameter(b)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out


class LoRALinear(Module):
    """Low-Rank Adaptation (LoRA) wrapper around a frozen ``Linear`` layer.

    Computes ``base(x) + ((x @ A^T) @ B^T) * (alpha / rank)`` where ``B`` is
    zero-initialized so step-0 output exactly matches the pretrained base layer.
    """

    def __init__(
        self,
        base: Linear,
        rank: int = 4,
        alpha: float = 8.0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.base = base
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)

        rng = _np.random.default_rng(seed) if seed is not None else _np.random.default_rng()
        a_init = rng.normal(0.0, 1.0 / math.sqrt(float(base.in_features)), size=(self.rank, base.in_features)).astype(_np.float32)
        b_init = _np.zeros((base.out_features, self.rank), dtype=_np.float32)
        self.lora_A = Parameter(a_init)
        self.lora_B = Parameter(b_init)

    def forward(self, x: Tensor) -> Tensor:
        base_out = self.base(x)
        lora_out = ((x @ self.lora_A.T) @ self.lora_B.T) * self.scaling
        return base_out + lora_out

    def merge_weights(self) -> Linear:
        """Fold learned LoRA update ``(alpha/rank) * (B @ A)`` into a plain ``Linear`` layer."""
        merged = Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        delta = (self.lora_B.data @ self.lora_A.data) * self.scaling
        merged.weight = Parameter((self.base.weight.data + delta).contiguous())
        if self.base.bias is not None:
            merged.bias = Parameter(self.base.bias.data.contiguous())
        return merged


class KVCache:
    """VRAM Key/Value cache for fast O(1)-per-step autoregressive Transformer decoding."""

    def __init__(self) -> None:
        self.k: Optional[vnp.ndarray] = None
        self.v: Optional[vnp.ndarray] = None

    @property
    def seq_len(self) -> int:
        return 0 if self.k is None else int(self.k.shape[2])

    def reset(self) -> None:
        self.k = None
        self.v = None

    def update(self, new_k: Tensor, new_v: Tensor) -> Tuple[Tensor, Tensor]:
        if self.k is None or self.v is None:
            self.k = new_k.data.contiguous()
            self.v = new_v.data.contiguous()
        else:
            self.k = vnp.concatenate([self.k, new_k.data.contiguous()], axis=2).contiguous()
            self.v = vnp.concatenate([self.v, new_v.data.contiguous()], axis=2).contiguous()
        return Tensor(self.k, requires_grad=False), Tensor(self.v, requires_grad=False)


class Conv2d(Module):
    """2D Convolution layer via ``im2col`` + Vulkan GPU GEMM."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)

        rng = _np.random.default_rng(seed) if seed is not None else _np.random.default_rng()
        fan_in = in_channels * self.kernel_size * self.kernel_size
        bound = 1.0 / math.sqrt(float(fan_in))
        w = rng.uniform(-bound, bound, size=(out_channels, in_channels, self.kernel_size, self.kernel_size)).astype(_np.float32)
        self.weight = Parameter(w)
        self.bias: Optional[Parameter] = Parameter(rng.uniform(-bound, bound, size=(out_channels,)).astype(_np.float32)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        return F.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)


class MaxPool2d(Module):
    def __init__(self, kernel_size: int = 2, stride: Optional[int] = None) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        return F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)


class Flatten(Module):
    def __init__(self, start_dim: int = 1) -> None:
        super().__init__()
        self.start_dim = int(start_dim)

    def forward(self, x: Tensor) -> Tensor:
        lead = x.shape[: self.start_dim]
        tail = 1
        for s in x.shape[self.start_dim :]:
            tail *= s
        return x.reshape(*lead, tail)


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class GELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.gelu()


class SiLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.silu()


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class Dropout(Module):
    def __init__(self, p: float = 0.1) -> None:
        super().__init__()
        self.p = float(p)

    def forward(self, x: Tensor) -> Tensor:
        return F.dropout(x, p=self.p, training=self.training)


class LayerNorm(Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = Parameter(_np.ones((int(normalized_shape),), dtype=_np.float32))
        self.bias = Parameter(_np.zeros((int(normalized_shape),), dtype=_np.float32))

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(x, self.weight, self.bias, eps=self.eps)


class RMSNorm(Module):
    """Root Mean Square Layer Normalization (used in Llama / Gemma / modern LLMs)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = Parameter(_np.ones((int(dim),), dtype=_np.float32))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, self.weight, eps=self.eps)


class SwiGLU(Module):
    """SwiGLU Feed-Forward Network: ``down(silu(gate(x)) * up(x))`` (modern LLM FFN)."""

    def __init__(self, dim: int, hidden_dim: int, seed: Optional[int] = None) -> None:
        super().__init__()
        self.w_gate = Linear(dim, hidden_dim, bias=False, seed=seed)
        self.w_up = Linear(dim, hidden_dim, bias=False, seed=None if seed is None else seed + 1)
        self.w_down = Linear(hidden_dim, dim, bias=False, seed=None if seed is None else seed + 2)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_down(F.swiglu(self.w_gate(x), self.w_up(x)))


class LlamaBlock(Module):
    """Modern Llama-style Transformer block: RMSNorm + Causal Attention (optional RoPE + KV-Cache) + RMSNorm + SwiGLU."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        is_causal: bool = True,
        use_rope: bool = False,
        rope_theta: float = 10000.0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        h_dim = int(hidden_dim) if hidden_dim is not None else int(embed_dim * 8 // 3)
        self.norm1 = RMSNorm(embed_dim)
        self.attn = MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, is_causal=is_causal,
            use_rope=use_rope, rope_theta=rope_theta, seed=seed,
        )
        self.norm2 = RMSNorm(embed_dim)
        self.ffn = SwiGLU(embed_dim, h_dim, seed=None if seed is None else seed + 10)

    def forward(self, x: Tensor, *, kv_cache: Optional[KVCache] = None, start_pos: Optional[int] = None) -> Tensor:
        x = x + self.attn(self.norm1(x), kv_cache=kv_cache, start_pos=start_pos)
        return x + self.ffn(self.norm2(x))


class BatchNorm1d(Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.weight = Parameter(_np.ones((self.num_features,), dtype=_np.float32))
        self.bias = Parameter(_np.zeros((self.num_features,), dtype=_np.float32))
        self._running_mean = vnp.zeros((1, self.num_features), dtype=_np.float32)
        self._running_var = vnp.ones((1, self.num_features), dtype=_np.float32)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mean = x.mean(axis=0, keepdims=True)
            diff = x - mean
            var = (diff * diff).mean(axis=0, keepdims=True)
            with no_grad():
                m = self.momentum
                self._running_mean = self._running_mean * (1.0 - m) + mean.data * m
                self._running_var = self._running_var * (1.0 - m) + var.data * m
            norm = diff / (var + self.eps).sqrt()
        else:
            rm = Tensor(self._running_mean, requires_grad=False)
            rv = Tensor(self._running_var, requires_grad=False)
            norm = (x - rm) / (rv + self.eps).sqrt()
        return norm * self.weight + self.bias


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, seed: Optional[int] = None) -> None:
        super().__init__()
        rng = _np.random.default_rng(seed) if seed is not None else _np.random.default_rng()
        w = rng.normal(0.0, 0.05, size=(int(num_embeddings), int(embedding_dim))).astype(_np.float32)
        self.weight = Parameter(w)

    def forward(self, indices) -> Tensor:
        return F.embedding(indices, self.weight)


class MultiheadAttention(Module):
    """Multi-Head Self-Attention on Vulkan GPU (supports causal masking, RoPE, & KV-Cache)."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_causal: bool = False,
        use_rope: bool = False,
        rope_theta: float = 10000.0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.is_causal = bool(is_causal)
        self.use_rope = bool(use_rope)
        self.rope_theta = float(rope_theta)
        self.q_proj = Linear(embed_dim, embed_dim, seed=seed)
        self.k_proj = Linear(embed_dim, embed_dim, seed=None if seed is None else seed + 1)
        self.v_proj = Linear(embed_dim, embed_dim, seed=None if seed is None else seed + 2)
        self.out_proj = Linear(embed_dim, embed_dim, seed=None if seed is None else seed + 3)
        self.drop = Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        *,
        is_causal: Optional[bool] = None,
        kv_cache: Optional[KVCache] = None,
        start_pos: Optional[int] = None,
    ) -> Tensor:
        B, T, C = x.shape
        H, D = self.num_heads, self.head_dim
        pos = (kv_cache.seq_len if kv_cache is not None else 0) if start_pos is None else int(start_pos)

        q = self.q_proj(x).reshape(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        k = self.k_proj(x).reshape(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, H, D).transpose(1, 2)

        if self.use_rope:
            q = F.apply_rotary_pos_emb(q, start_pos=pos, theta=self.rope_theta)
            k = F.apply_rotary_pos_emb(k, start_pos=pos, theta=self.rope_theta)

        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        causal = self.is_causal if is_causal is None else bool(is_causal)
        ctx = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        ctx = ctx.transpose(1, 2).reshape(B, T, C)
        return self.drop(self.out_proj(ctx))


class TransformerBlock(Module):
    """Pre-LN Transformer block (MultiheadAttention + GELU MLP with residual connections)."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0, is_causal: bool = False, seed: Optional[int] = None) -> None:
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.ln1 = LayerNorm(embed_dim)
        self.attn = MultiheadAttention(embed_dim, num_heads, dropout=dropout, is_causal=is_causal, seed=seed)
        self.ln2 = LayerNorm(embed_dim)
        self.fc1 = Linear(embed_dim, hidden, seed=None if seed is None else seed + 10)
        self.fc2 = Linear(hidden, embed_dim, seed=None if seed is None else seed + 11)
        self.drop = Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        h = self.fc2(self.fc1(self.ln2(x)).gelu())
        return x + self.drop(h)


# --- Loss Modules ---
class MSELoss(Module):
    def forward(self, pred: Tensor, target) -> Tensor:
        return F.mse_loss(pred, target)


class CrossEntropyLoss(Module):
    def forward(self, logits: Tensor, targets) -> Tensor:
        return F.cross_entropy(logits, targets)


class BCEWithLogitsLoss(Module):
    def forward(self, logits: Tensor, targets) -> Tensor:
        return F.binary_cross_entropy_with_logits(logits, targets)
