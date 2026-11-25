"""vktorch._kernels: Fused single-pass Vulkan SPIR-V kernels for AdamW, SGD, and GELU."""

from __future__ import annotations

import struct
from typing import Dict

import numpy as _np
import vknumpy as vnp
from vknumpy.kernel import _compile_glsl_to_spirv
from vkspatial._backend import dispatch_spv_gpu

_SPV_CACHE: Dict[str, bytes] = {}

_GLSL_ADAMW = """#version 450
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer ParamBuf { float param[]; };
layout(set = 0, binding = 1) readonly buffer GradBuf { float grad[]; };
layout(set = 0, binding = 2) buffer MBuf { float m[]; };
layout(set = 0, binding = 3) buffer VBuf { float v[]; };

layout(push_constant) uniform PC {
    int   n;
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    float inv_bc1;
    float inv_sqrt_bc2;
    int   decoupled;
} pc;

void main() {
    int i = int(gl_GlobalInvocationID.x);
    if (i >= pc.n) return;

    float p = param[i];
    float g = grad[i];
    if (pc.decoupled != 0) {
        p *= (1.0 - pc.lr * pc.weight_decay);
    } else if (pc.weight_decay != 0.0) {
        g += p * pc.weight_decay;
    }

    float mi = pc.beta1 * m[i] + (1.0 - pc.beta1) * g;
    float vi = pc.beta2 * v[i] + (1.0 - pc.beta2) * (g * g);
    m[i] = mi;
    v[i] = vi;

    float step = (mi * pc.inv_bc1) / (sqrt(vi) * pc.inv_sqrt_bc2 + pc.eps);
    param[i] = p - pc.lr * step;
}
"""

_GLSL_SGD = """#version 450
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer ParamBuf { float param[]; };
layout(set = 0, binding = 1) readonly buffer GradBuf { float grad[]; };
layout(set = 0, binding = 2) buffer VelBuf { float vel[]; };

layout(push_constant) uniform PC {
    int   n;
    float lr;
    float momentum;
    float weight_decay;
} pc;

void main() {
    int i = int(gl_GlobalInvocationID.x);
    if (i >= pc.n) return;

    float p = param[i];
    float g = grad[i];
    if (pc.weight_decay != 0.0) g += p * pc.weight_decay;
    if (pc.momentum != 0.0) {
        float v = pc.momentum * vel[i] + g;
        vel[i] = v;
        g = v;
    }
    param[i] = p - pc.lr * g;
}
"""

_GLSL_GELU_FWD = """#version 450
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer InBuf  { float x_in[]; };
layout(set = 0, binding = 1) writeonly buffer OutBuf { float y_out[]; };

layout(push_constant) uniform PC { int n; } pc;

void main() {
    int i = int(gl_GlobalInvocationID.x);
    if (i >= pc.n) return;
    float x = x_in[i];
    float c = 0.7978845608; // sqrt(2 / pi)
    float inner = c * (x + 0.044715 * x * x * x);
    y_out[i] = 0.5 * x * (1.0 + tanh(inner));
}
"""

_GLSL_GELU_BWD = """#version 450
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer InBuf   { float x_in[]; };
layout(set = 0, binding = 1) readonly buffer GOutBuf { float g_out[]; };
layout(set = 0, binding = 2) writeonly buffer GInBuf { float g_in[]; };

layout(push_constant) uniform PC { int n; } pc;

void main() {
    int i = int(gl_GlobalInvocationID.x);
    if (i >= pc.n) return;
    float x = x_in[i];
    float c = 0.7978845608;
    float inner = c * (x + 0.044715 * x * x * x);
    float th = tanh(inner);
    float sech2 = 1.0 - th * th;
    float d_inner = c * (1.0 + 3.0 * 0.044715 * x * x);
    float d = 0.5 * (1.0 + th) + 0.5 * x * sech2 * d_inner;
    g_in[i] = g_out[i] * d;
}
"""


def _get_spv(key: str, glsl: str) -> bytes:
    spv = _SPV_CACHE.get(key)
    if spv is None:
        spv = _compile_glsl_to_spirv(glsl)
        _SPV_CACHE[key] = spv
    return spv


def fused_adam_step(
    param: vnp.ndarray,
    grad: vnp.ndarray,
    m: vnp.ndarray,
    v: vnp.ndarray,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    inv_bc1: float,
    inv_sqrt_bc2: float,
    decoupled: bool = True,
) -> None:
    """In-place single-pass Vulkan SPIR-V Adam / AdamW update on VRAM buffers."""
    n = int(param.size)
    if n == 0:
        return
    spv = _get_spv("adamw", _GLSL_ADAMW)
    pc = struct.pack(
        "<ifffffffi",
        n, float(lr), float(beta1), float(beta2), float(eps),
        float(weight_decay), float(inv_bc1), float(inv_sqrt_bc2),
        1 if decoupled else 0,
    )
    dispatch_spv_gpu(spv, [param, grad, m, v], pc, ((n + 255) // 256, 1, 1))


def fused_sgd_step(
    param: vnp.ndarray,
    grad: vnp.ndarray,
    vel: vnp.ndarray,
    *,
    lr: float,
    momentum: float,
    weight_decay: float,
) -> None:
    """In-place single-pass Vulkan SPIR-V SGD + momentum + weight decay update."""
    n = int(param.size)
    if n == 0:
        return
    spv = _get_spv("sgd", _GLSL_SGD)
    pc = struct.pack("<ifff", n, float(lr), float(momentum), float(weight_decay))
    dispatch_spv_gpu(spv, [param, grad, vel], pc, ((n + 255) // 256, 1, 1))


def fused_gelu_fwd(x: vnp.ndarray) -> vnp.ndarray:
    xc = x.contiguous()
    out = vnp.ndarray._empty_uninit(xc.shape, _np.float32)
    n = int(xc.size)
    if n > 0:
        spv = _get_spv("gelu_fwd", _GLSL_GELU_FWD)
        dispatch_spv_gpu(spv, [xc, out], struct.pack("<i", n), ((n + 255) // 256, 1, 1))
    return out


def fused_gelu_bwd(x: vnp.ndarray, g_out: vnp.ndarray) -> vnp.ndarray:
    xc = x.contiguous()
    gc = g_out.contiguous()
    g_in = vnp.ndarray._empty_uninit(xc.shape, _np.float32)
    n = int(xc.size)
    if n > 0:
        spv = _get_spv("gelu_bwd", _GLSL_GELU_BWD)
        dispatch_spv_gpu(spv, [xc, gc, g_in], struct.pack("<i", n), ((n + 255) // 256, 1, 1))
    return g_in
