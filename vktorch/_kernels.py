"""vktorch._kernels: Fused Vulkan SPIR-V GPU kernels for optimizers, activations, norms, softmax, and cross-entropy."""

from __future__ import annotations

import struct
from typing import Dict, Optional, Tuple

import numpy as _np
import vknumpy as vnp
from vknumpy.kernel import _compile_glsl_to_spirv
from vkspatial._backend import dispatch_spv_gpu

_SPV_CACHE: Dict[str, bytes] = {}
_DUMMY_F32: Optional[vnp.ndarray] = None


def _dummy_f32() -> vnp.ndarray:
    global _DUMMY_F32
    if _DUMMY_F32 is None:
        _DUMMY_F32 = vnp.zeros((1,), dtype=_np.float32)
    return _DUMMY_F32


# =====================================================================
# 1. Optimizers: Fused Adam / AdamW & SGD
# =====================================================================

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

# =====================================================================
# 2. Activations: Fused GELU, SiLU, and SwiGLU (silu(gate) * up)
# =====================================================================

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

_GLSL_SILU_FWD = """#version 450
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer InBuf   { float x_in[]; };
layout(set = 0, binding = 1) writeonly buffer OutBuf { float y_out[]; };

layout(push_constant) uniform PC { int n; } pc;

void main() {
    int i = int(gl_GlobalInvocationID.x);
    if (i >= pc.n) return;
    float x = x_in[i];
    float sig = 1.0 / (1.0 + exp(-clamp(x, -30.0, 30.0)));
    y_out[i] = x * sig;
}
"""

_GLSL_SILU_BWD = """#version 450
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer InBuf   { float x_in[]; };
layout(set = 0, binding = 1) readonly buffer GOutBuf { float g_out[]; };
layout(set = 0, binding = 2) writeonly buffer GInBuf { float g_in[]; };

layout(push_constant) uniform PC { int n; } pc;

void main() {
    int i = int(gl_GlobalInvocationID.x);
    if (i >= pc.n) return;
    float x = x_in[i];
    float sig = 1.0 / (1.0 + exp(-clamp(x, -30.0, 30.0)));
    float d = sig * (1.0 + x * (1.0 - sig));
    g_in[i] = g_out[i] * d;
}
"""

_GLSL_SWIGLU_FWD = """#version 450
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer GateBuf { float gate[]; };
layout(set = 0, binding = 1) readonly buffer UpBuf   { float up[]; };
layout(set = 0, binding = 2) writeonly buffer OutBuf { float y_out[]; };

layout(push_constant) uniform PC { int n; } pc;

void main() {
    int i = int(gl_GlobalInvocationID.x);
    if (i >= pc.n) return;
    float g = gate[i];
    float sig = 1.0 / (1.0 + exp(-clamp(g, -30.0, 30.0)));
    y_out[i] = (g * sig) * up[i];
}
"""

_GLSL_SWIGLU_BWD = """#version 450
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer GateBuf  { float gate[]; };
layout(set = 0, binding = 1) readonly buffer UpBuf    { float up[]; };
layout(set = 0, binding = 2) readonly buffer GOutBuf  { float g_out[]; };
layout(set = 0, binding = 3) writeonly buffer DGateBuf { float d_gate[]; };
layout(set = 0, binding = 4) writeonly buffer DUpBuf   { float d_up[]; };

layout(push_constant) uniform PC { int n; } pc;

void main() {
    int i = int(gl_GlobalInvocationID.x);
    if (i >= pc.n) return;
    float g = gate[i];
    float u = up[i];
    float go = g_out[i];
    float sig = 1.0 / (1.0 + exp(-clamp(g, -30.0, 30.0)));
    float silu_g = g * sig;
    float d_silu = sig * (1.0 + g * (1.0 - sig));
    d_up[i] = go * silu_g;
    d_gate[i] = go * u * d_silu;
}
"""

# =====================================================================
# 3. RMSNorm: Workgroup-reduced forward + exact analytical backward
# =====================================================================

_GLSL_RMSNORM_FWD = """#version 450
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer XBuf      { float x[]; };
layout(set = 0, binding = 1) readonly buffer WBuf      { float w[]; };
layout(set = 0, binding = 2) writeonly buffer YBuf     { float y[]; };
layout(set = 0, binding = 3) writeonly buffer InvRmsBuf { float inv_rms[]; };

layout(push_constant) uniform PC {
    int   n_rows;
    int   d;
    float eps;
} pc;

shared float sdata[256];

void main() {
    int row = int(gl_WorkGroupID.x);
    if (row >= pc.n_rows) return;
    int tid = int(gl_LocalInvocationID.x);
    int base = row * pc.d;

    float acc = 0.0;
    for (int c = tid; c < pc.d; c += 256) {
        float v = x[base + c];
        acc += v * v;
    }
    sdata[tid] = acc;
    barrier();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        barrier();
    }

    float ir = inversesqrt(sdata[0] / float(pc.d) + pc.eps);
    if (tid == 0) inv_rms[row] = ir;

    for (int c = tid; c < pc.d; c += 256) {
        y[base + c] = x[base + c] * ir * w[c];
    }
}
"""

_GLSL_RMSNORM_BWD_DX = """#version 450
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer XBuf      { float x[]; };
layout(set = 0, binding = 1) readonly buffer WBuf      { float w[]; };
layout(set = 0, binding = 2) readonly buffer InvRmsBuf { float inv_rms[]; };
layout(set = 0, binding = 3) readonly buffer GOutBuf   { float g_out[]; };
layout(set = 0, binding = 4) writeonly buffer DXBuf    { float dx[]; };

layout(push_constant) uniform PC {
    int n_rows;
    int d;
} pc;

shared float sdata[256];

void main() {
    int row = int(gl_WorkGroupID.x);
    if (row >= pc.n_rows) return;
    int tid = int(gl_LocalInvocationID.x);
    int base = row * pc.d;
    float ir = inv_rms[row];

    float acc = 0.0;
    for (int c = tid; c < pc.d; c += 256) {
        acc += g_out[base + c] * w[c] * x[base + c];
    }
    sdata[tid] = acc;
    barrier();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        barrier();
    }

    float coeff = (ir * ir / float(pc.d)) * sdata[0];
    for (int c = tid; c < pc.d; c += 256) {
        dx[base + c] = ir * (g_out[base + c] * w[c] - x[base + c] * coeff);
    }
}
"""

_GLSL_RMSNORM_BWD_DW = """#version 450
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer XBuf      { float x[]; };
layout(set = 0, binding = 1) readonly buffer InvRmsBuf { float inv_rms[]; };
layout(set = 0, binding = 2) readonly buffer GOutBuf   { float g_out[]; };
layout(set = 0, binding = 3) writeonly buffer DWBuf    { float dw[]; };

layout(push_constant) uniform PC {
    int n_rows;
    int d;
} pc;

shared float sdata[256];

void main() {
    int col = int(gl_WorkGroupID.x);
    if (col >= pc.d) return;
    int tid = int(gl_LocalInvocationID.x);

    float acc = 0.0;
    for (int r = tid; r < pc.n_rows; r += 256) {
        int idx = r * pc.d + col;
        acc += g_out[idx] * x[idx] * inv_rms[r];
    }
    sdata[tid] = acc;
    barrier();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        barrier();
    }

    if (tid == 0) dw[col] = sdata[0];
}
"""

# =====================================================================
# 4. LayerNorm: Workgroup-reduced forward + exact analytical backward
# =====================================================================

_GLSL_LAYERNORM_FWD = """#version 450
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer XBuf      { float x[]; };
layout(set = 0, binding = 1) readonly buffer WBuf      { float w[]; };
layout(set = 0, binding = 2) readonly buffer BBuf      { float b[]; };
layout(set = 0, binding = 3) writeonly buffer YBuf     { float y[]; };
layout(set = 0, binding = 4) writeonly buffer MeanBuf  { float mean_out[]; };
layout(set = 0, binding = 5) writeonly buffer InvStdBuf { float inv_std_out[]; };

layout(push_constant) uniform PC {
    int   n_rows;
    int   d;
    float eps;
    int   has_w;
    int   has_b;
} pc;

shared float sdata[256];

void main() {
    int row = int(gl_WorkGroupID.x);
    if (row >= pc.n_rows) return;
    int tid = int(gl_LocalInvocationID.x);
    int base = row * pc.d;

    float sum_x = 0.0;
    for (int c = tid; c < pc.d; c += 256) {
        sum_x += x[base + c];
    }
    sdata[tid] = sum_x;
    barrier();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        barrier();
    }
    float mu = sdata[0] / float(pc.d);
    barrier();

    float sum_sq = 0.0;
    for (int c = tid; c < pc.d; c += 256) {
        float diff = x[base + c] - mu;
        sum_sq += diff * diff;
    }
    sdata[tid] = sum_sq;
    barrier();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        barrier();
    }
    float inv_std = inversesqrt(sdata[0] / float(pc.d) + pc.eps);

    if (tid == 0) {
        mean_out[row] = mu;
        inv_std_out[row] = inv_std;
    }

    for (int c = tid; c < pc.d; c += 256) {
        float val = (x[base + c] - mu) * inv_std;
        if (pc.has_w != 0) val *= w[c];
        if (pc.has_b != 0) val += b[c];
        y[base + c] = val;
    }
}
"""

_GLSL_LAYERNORM_BWD_DX = """#version 450
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer XBuf      { float x[]; };
layout(set = 0, binding = 1) readonly buffer WBuf      { float w[]; };
layout(set = 0, binding = 2) readonly buffer MeanBuf   { float mean_in[]; };
layout(set = 0, binding = 3) readonly buffer InvStdBuf { float inv_std_in[]; };
layout(set = 0, binding = 4) readonly buffer GOutBuf   { float g_out[]; };
layout(set = 0, binding = 5) writeonly buffer DXBuf    { float dx[]; };

layout(push_constant) uniform PC {
    int n_rows;
    int d;
    int has_w;
} pc;

shared float s1[256];
shared float s2[256];

void main() {
    int row = int(gl_WorkGroupID.x);
    if (row >= pc.n_rows) return;
    int tid = int(gl_LocalInvocationID.x);
    int base = row * pc.d;
    float mu = mean_in[row];
    float istd = inv_std_in[row];

    float acc_dy = 0.0;
    float acc_dy_xh = 0.0;
    for (int c = tid; c < pc.d; c += 256) {
        float dy = g_out[base + c] * (pc.has_w != 0 ? w[c] : 1.0);
        float xh = (x[base + c] - mu) * istd;
        acc_dy += dy;
        acc_dy_xh += dy * xh;
    }
    s1[tid] = acc_dy;
    s2[tid] = acc_dy_xh;
    barrier();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            s1[tid] += s1[tid + s];
            s2[tid] += s2[tid + s];
        }
        barrier();
    }

    float inv_d = 1.0 / float(pc.d);
    float mean_dy = s1[0] * inv_d;
    float mean_dy_xh = s2[0] * inv_d;

    for (int c = tid; c < pc.d; c += 256) {
        float dy = g_out[base + c] * (pc.has_w != 0 ? w[c] : 1.0);
        float xh = (x[base + c] - mu) * istd;
        dx[base + c] = istd * (dy - mean_dy - xh * mean_dy_xh);
    }
}
"""

_GLSL_LAYERNORM_BWD_DW_DB = """#version 450
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer XBuf      { float x[]; };
layout(set = 0, binding = 1) readonly buffer MeanBuf   { float mean_in[]; };
layout(set = 0, binding = 2) readonly buffer InvStdBuf { float inv_std_in[]; };
layout(set = 0, binding = 3) readonly buffer GOutBuf   { float g_out[]; };
layout(set = 0, binding = 4) writeonly buffer DWBuf    { float dw[]; };
layout(set = 0, binding = 5) writeonly buffer DBBuf    { float db[]; };

layout(push_constant) uniform PC {
    int n_rows;
    int d;
    int need_dw;
    int need_db;
} pc;

shared float s_dw[256];
shared float s_db[256];

void main() {
    int col = int(gl_WorkGroupID.x);
    if (col >= pc.d) return;
    int tid = int(gl_LocalInvocationID.x);

    float acc_w = 0.0;
    float acc_b = 0.0;
    for (int r = tid; r < pc.n_rows; r += 256) {
        int idx = r * pc.d + col;
        float go = g_out[idx];
        if (pc.need_dw != 0) {
            float xh = (x[idx] - mean_in[r]) * inv_std_in[r];
            acc_w += go * xh;
        }
        if (pc.need_db != 0) {
            acc_b += go;
        }
    }
    s_dw[tid] = acc_w;
    s_db[tid] = acc_b;
    barrier();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            s_dw[tid] += s_dw[tid + s];
            s_db[tid] += s_db[tid + s];
        }
        barrier();
    }

    if (tid == 0) {
        if (pc.need_dw != 0) dw[col] = s_dw[0];
        if (pc.need_db != 0) db[col] = s_db[0];
    }
}
"""

# =====================================================================
# 5. Softmax (last axis): Workgroup-reduced forward & backward
# =====================================================================

_GLSL_SOFTMAX_FWD = """#version 450
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer XBuf  { float x[]; };
layout(set = 0, binding = 1) writeonly buffer YBuf { float y[]; };

layout(push_constant) uniform PC {
    int n_rows;
    int d;
} pc;

shared float sdata[256];

void main() {
    int row = int(gl_WorkGroupID.x);
    if (row >= pc.n_rows) return;
    int tid = int(gl_LocalInvocationID.x);
    int base = row * pc.d;

    float local_max = -3.402823466e38;
    for (int c = tid; c < pc.d; c += 256) {
        local_max = max(local_max, x[base + c]);
    }
    sdata[tid] = local_max;
    barrier();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = max(sdata[tid], sdata[tid + s]);
        barrier();
    }
    float row_max = sdata[0];
    barrier();

    float local_sum = 0.0;
    for (int c = tid; c < pc.d; c += 256) {
        local_sum += exp(x[base + c] - row_max);
    }
    sdata[tid] = local_sum;
    barrier();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        barrier();
    }
    float inv_sum = 1.0 / sdata[0];

    for (int c = tid; c < pc.d; c += 256) {
        y[base + c] = exp(x[base + c] - row_max) * inv_sum;
    }
}
"""

_GLSL_SOFTMAX_BWD = """#version 450
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer SmBuf   { float sm[]; };
layout(set = 0, binding = 1) readonly buffer GOutBuf { float g_out[]; };
layout(set = 0, binding = 2) writeonly buffer DXBuf  { float dx[]; };

layout(push_constant) uniform PC {
    int n_rows;
    int d;
} pc;

shared float sdata[256];

void main() {
    int row = int(gl_WorkGroupID.x);
    if (row >= pc.n_rows) return;
    int tid = int(gl_LocalInvocationID.x);
    int base = row * pc.d;

    float dot_acc = 0.0;
    for (int c = tid; c < pc.d; c += 256) {
        dot_acc += g_out[base + c] * sm[base + c];
    }
    sdata[tid] = dot_acc;
    barrier();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        barrier();
    }
    float dot_val = sdata[0];

    for (int c = tid; c < pc.d; c += 256) {
        dx[base + c] = sm[base + c] * (g_out[base + c] - dot_val);
    }
}
"""

# =====================================================================
# 6. Cross-Entropy: Single-pass fused loss + optional exact dlogits
# =====================================================================

_GLSL_CROSS_ENTROPY = """#version 450
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer LogitsBuf  { float logits[]; };
layout(set = 0, binding = 1) readonly buffer TargetsBuf { int targets[]; };
layout(set = 0, binding = 2) writeonly buffer LossesBuf { float losses[]; };
layout(set = 0, binding = 3) writeonly buffer DLogitsBuf { float dlogits[]; };

layout(push_constant) uniform PC {
    int n_rows;
    int c_dim;
    int write_grad;
} pc;

shared float sdata[256];

void main() {
    int row = int(gl_WorkGroupID.x);
    if (row >= pc.n_rows) return;
    int tid = int(gl_LocalInvocationID.x);
    int base = row * pc.c_dim;
    int tgt = targets[row];

    float local_max = -3.402823466e38;
    for (int c = tid; c < pc.c_dim; c += 256) {
        local_max = max(local_max, logits[base + c]);
    }
    sdata[tid] = local_max;
    barrier();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = max(sdata[tid], sdata[tid + s]);
        barrier();
    }
    float row_max = sdata[0];
    barrier();

    float local_sum = 0.0;
    for (int c = tid; c < pc.c_dim; c += 256) {
        local_sum += exp(logits[base + c] - row_max);
    }
    sdata[tid] = local_sum;
    barrier();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        barrier();
    }
    float sum_ex = sdata[0];
    float log_sum_ex = log(sum_ex);

    if (tid == 0) {
        losses[row] = (row_max + log_sum_ex) - logits[base + tgt];
    }

    if (pc.write_grad != 0) {
        float inv_n = 1.0 / float(pc.n_rows);
        float inv_sum = 1.0 / sum_ex;
        for (int c = tid; c < pc.c_dim; c += 256) {
            float p = exp(logits[base + c] - row_max) * inv_sum;
            if (c == tgt) p -= 1.0;
            dlogits[base + c] = p * inv_n;
        }
    }
}
"""


def _get_spv(key: str, glsl: str) -> bytes:
    spv = _SPV_CACHE.get(key)
    if spv is None:
        spv = _compile_glsl_to_spirv(glsl)
        _SPV_CACHE[key] = spv
    return spv


# =====================================================================
# Python dispatch wrappers
# =====================================================================

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


def fused_silu_fwd(x: vnp.ndarray) -> vnp.ndarray:
    xc = x.contiguous()
    out = vnp.ndarray._empty_uninit(xc.shape, _np.float32)
    n = int(xc.size)
    if n > 0:
        spv = _get_spv("silu_fwd", _GLSL_SILU_FWD)
        dispatch_spv_gpu(spv, [xc, out], struct.pack("<i", n), ((n + 255) // 256, 1, 1))
    return out


def fused_silu_bwd(x: vnp.ndarray, g_out: vnp.ndarray) -> vnp.ndarray:
    xc = x.contiguous()
    gc = g_out.contiguous()
    g_in = vnp.ndarray._empty_uninit(xc.shape, _np.float32)
    n = int(xc.size)
    if n > 0:
        spv = _get_spv("silu_bwd", _GLSL_SILU_BWD)
        dispatch_spv_gpu(spv, [xc, gc, g_in], struct.pack("<i", n), ((n + 255) // 256, 1, 1))
    return g_in


def fused_swiglu_fwd(gate: vnp.ndarray, up: vnp.ndarray) -> vnp.ndarray:
    gc = gate.contiguous()
    uc = up.contiguous()
    out = vnp.ndarray._empty_uninit(gc.shape, _np.float32)
    n = int(gc.size)
    if n > 0:
        spv = _get_spv("swiglu_fwd", _GLSL_SWIGLU_FWD)
        dispatch_spv_gpu(spv, [gc, uc, out], struct.pack("<i", n), ((n + 255) // 256, 1, 1))
    return out


def fused_swiglu_bwd(
    gate: vnp.ndarray, up: vnp.ndarray, g_out: vnp.ndarray
) -> Tuple[vnp.ndarray, vnp.ndarray]:
    gc = gate.contiguous()
    uc = up.contiguous()
    goc = g_out.contiguous()
    d_gate = vnp.ndarray._empty_uninit(gc.shape, _np.float32)
    d_up = vnp.ndarray._empty_uninit(uc.shape, _np.float32)
    n = int(gc.size)
    if n > 0:
        spv = _get_spv("swiglu_bwd", _GLSL_SWIGLU_BWD)
        dispatch_spv_gpu(spv, [gc, uc, goc, d_gate, d_up], struct.pack("<i", n), ((n + 255) // 256, 1, 1))
    return d_gate, d_up


def fused_rmsnorm_fwd(
    x: vnp.ndarray, weight: vnp.ndarray, eps: float = 1e-6
) -> Tuple[vnp.ndarray, vnp.ndarray]:
    xc = x.contiguous()
    wc = weight.contiguous()
    d = int(xc.shape[-1])
    n_rows = int(xc.size) // d if d > 0 else 0
    y = vnp.ndarray._empty_uninit(xc.shape, _np.float32)
    inv_rms = vnp.ndarray._empty_uninit((n_rows,), _np.float32)
    if n_rows > 0 and d > 0:
        spv = _get_spv("rmsnorm_fwd", _GLSL_RMSNORM_FWD)
        pc = struct.pack("<iif", n_rows, d, float(eps))
        dispatch_spv_gpu(spv, [xc, wc, y, inv_rms], pc, (n_rows, 1, 1))
    return y, inv_rms


def fused_rmsnorm_bwd(
    x: vnp.ndarray,
    weight: vnp.ndarray,
    inv_rms: vnp.ndarray,
    g_out: vnp.ndarray,
    *,
    need_dx: bool = True,
    need_dw: bool = True,
) -> Tuple[Optional[vnp.ndarray], Optional[vnp.ndarray]]:
    xc = x.contiguous()
    wc = weight.contiguous()
    gc = g_out.contiguous()
    d = int(xc.shape[-1])
    n_rows = int(xc.size) // d if d > 0 else 0
    dx: Optional[vnp.ndarray] = None
    dw: Optional[vnp.ndarray] = None
    if n_rows > 0 and d > 0:
        pc = struct.pack("<ii", n_rows, d)
        if need_dx:
            dx = vnp.ndarray._empty_uninit(xc.shape, _np.float32)
            spv_dx = _get_spv("rmsnorm_bwd_dx", _GLSL_RMSNORM_BWD_DX)
            dispatch_spv_gpu(spv_dx, [xc, wc, inv_rms, gc, dx], pc, (n_rows, 1, 1))
        if need_dw:
            dw = vnp.ndarray._empty_uninit((d,), _np.float32)
            spv_dw = _get_spv("rmsnorm_bwd_dw", _GLSL_RMSNORM_BWD_DW)
            dispatch_spv_gpu(spv_dw, [xc, inv_rms, gc, dw], pc, (d, 1, 1))
    return dx, dw


def fused_layernorm_fwd(
    x: vnp.ndarray,
    weight: Optional[vnp.ndarray] = None,
    bias: Optional[vnp.ndarray] = None,
    eps: float = 1e-5,
) -> Tuple[vnp.ndarray, vnp.ndarray, vnp.ndarray]:
    xc = x.contiguous()
    wc = weight.contiguous() if weight is not None else _dummy_f32()
    bc = bias.contiguous() if bias is not None else _dummy_f32()
    d = int(xc.shape[-1])
    n_rows = int(xc.size) // d if d > 0 else 0
    y = vnp.ndarray._empty_uninit(xc.shape, _np.float32)
    mean = vnp.ndarray._empty_uninit((n_rows,), _np.float32)
    inv_std = vnp.ndarray._empty_uninit((n_rows,), _np.float32)
    if n_rows > 0 and d > 0:
        spv = _get_spv("layernorm_fwd", _GLSL_LAYERNORM_FWD)
        pc = struct.pack(
            "<iifii", n_rows, d, float(eps),
            1 if weight is not None else 0,
            1 if bias is not None else 0,
        )
        dispatch_spv_gpu(spv, [xc, wc, bc, y, mean, inv_std], pc, (n_rows, 1, 1))
    return y, mean, inv_std


def fused_layernorm_bwd(
    x: vnp.ndarray,
    weight: Optional[vnp.ndarray],
    mean: vnp.ndarray,
    inv_std: vnp.ndarray,
    g_out: vnp.ndarray,
    *,
    need_dx: bool = True,
    need_dw: bool = True,
    need_db: bool = True,
) -> Tuple[Optional[vnp.ndarray], Optional[vnp.ndarray], Optional[vnp.ndarray]]:
    xc = x.contiguous()
    wc = weight.contiguous() if weight is not None else _dummy_f32()
    gc = g_out.contiguous()
    d = int(xc.shape[-1])
    n_rows = int(xc.size) // d if d > 0 else 0
    dx: Optional[vnp.ndarray] = None
    dw: Optional[vnp.ndarray] = None
    db: Optional[vnp.ndarray] = None
    if n_rows > 0 and d > 0:
        if need_dx:
            dx = vnp.ndarray._empty_uninit(xc.shape, _np.float32)
            spv_dx = _get_spv("layernorm_bwd_dx", _GLSL_LAYERNORM_BWD_DX)
            pc_dx = struct.pack("<iii", n_rows, d, 1 if weight is not None else 0)
            dispatch_spv_gpu(spv_dx, [xc, wc, mean, inv_std, gc, dx], pc_dx, (n_rows, 1, 1))
        if need_dw or need_db:
            dw_buf = vnp.ndarray._empty_uninit((d,), _np.float32) if need_dw else _dummy_f32()
            db_buf = vnp.ndarray._empty_uninit((d,), _np.float32) if need_db else _dummy_f32()
            spv_dw = _get_spv("layernorm_bwd_dw_db", _GLSL_LAYERNORM_BWD_DW_DB)
            pc_dw = struct.pack("<iiii", n_rows, d, 1 if need_dw else 0, 1 if need_db else 0)
            dispatch_spv_gpu(spv_dw, [xc, mean, inv_std, gc, dw_buf, db_buf], pc_dw, (d, 1, 1))
            if need_dw:
                dw = dw_buf
            if need_db:
                db = db_buf
    return dx, dw, db


def fused_softmax_fwd(x: vnp.ndarray) -> vnp.ndarray:
    xc = x.contiguous()
    d = int(xc.shape[-1])
    n_rows = int(xc.size) // d if d > 0 else 0
    y = vnp.ndarray._empty_uninit(xc.shape, _np.float32)
    if n_rows > 0 and d > 0:
        spv = _get_spv("softmax_fwd", _GLSL_SOFTMAX_FWD)
        dispatch_spv_gpu(spv, [xc, y], struct.pack("<ii", n_rows, d), (n_rows, 1, 1))
    return y


def fused_softmax_bwd(sm: vnp.ndarray, g_out: vnp.ndarray) -> vnp.ndarray:
    smc = sm.contiguous()
    gc = g_out.contiguous()
    d = int(smc.shape[-1])
    n_rows = int(smc.size) // d if d > 0 else 0
    dx = vnp.ndarray._empty_uninit(smc.shape, _np.float32)
    if n_rows > 0 and d > 0:
        spv = _get_spv("softmax_bwd", _GLSL_SOFTMAX_BWD)
        dispatch_spv_gpu(spv, [smc, gc, dx], struct.pack("<ii", n_rows, d), (n_rows, 1, 1))
    return dx


def fused_cross_entropy(
    logits: vnp.ndarray, targets_i32: vnp.ndarray, *, compute_grad: bool = True
) -> Tuple[vnp.ndarray, Optional[vnp.ndarray]]:
    """Single-pass Vulkan SPIR-V CrossEntropy: returns ``(mean_loss_scalar, dlogits_or_none)``."""
    lc = logits.contiguous()
    tc = targets_i32.contiguous()
    c_dim = int(lc.shape[-1])
    n_rows = int(lc.size) // c_dim if c_dim > 0 else 0
    losses = vnp.ndarray._empty_uninit((n_rows,), _np.float32)
    dlogits = vnp.ndarray._empty_uninit(lc.shape, _np.float32) if compute_grad else _dummy_f32()
    if n_rows > 0 and c_dim > 0:
        spv = _get_spv("cross_entropy", _GLSL_CROSS_ENTROPY)
        pc = struct.pack("<iii", n_rows, c_dim, 1 if compute_grad else 0)
        dispatch_spv_gpu(spv, [lc, tc, losses, dlogits], pc, (n_rows, 1, 1))
    loss_val = vnp.mean(losses)
    return loss_val, (dlogits if compute_grad else None)


# =====================================================================
# 7. Vision: Fused SPIR-V im2col, col2im, & MaxPool2d (zero CPU copies)
# =====================================================================

_GLSL_IM2COL = """#version 450
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer XBuf    { float x[]; };
layout(set = 0, binding = 1) writeonly buffer ColBuf { float cols[]; };

layout(push_constant) uniform PC {
    int n;
    int c;
    int h;
    int w;
    int kh;
    int kw;
    int stride;
    int padding;
    int oh;
    int ow;
} pc;

void main() {
    int k_dim = pc.c * pc.kh * pc.kw;
    int total = pc.n * pc.oh * pc.ow * k_dim;
    int idx = int(gl_GlobalInvocationID.x);
    if (idx >= total) return;

    int row = idx / k_dim;
    int col = idx - row * k_dim;

    int ow_i = row % pc.ow;
    int tmp = row / pc.ow;
    int oh_i = tmp % pc.oh;
    int n_i  = tmp / pc.oh;

    int kw_i = col % pc.kw;
    int tmp2 = col / pc.kw;
    int kh_i = tmp2 % pc.kh;
    int c_i  = tmp2 / pc.kh;

    int ih = oh_i * pc.stride - pc.padding + kh_i;
    int iw = ow_i * pc.stride - pc.padding + kw_i;

    if (ih >= 0 && ih < pc.h && iw >= 0 && iw < pc.w) {
        cols[idx] = x[((n_i * pc.c + c_i) * pc.h + ih) * pc.w + iw];
    } else {
        cols[idx] = 0.0;
    }
}
"""

_GLSL_COL2IM = """#version 450
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer DColBuf { float dcols[]; };
layout(set = 0, binding = 1) writeonly buffer DXBuf  { float dx[]; };

layout(push_constant) uniform PC {
    int n;
    int c;
    int h;
    int w;
    int kh;
    int kw;
    int stride;
    int padding;
    int oh;
    int ow;
} pc;

void main() {
    int total = pc.n * pc.c * pc.h * pc.w;
    int idx = int(gl_GlobalInvocationID.x);
    if (idx >= total) return;

    int iw = idx % pc.w;
    int tmp = idx / pc.w;
    int ih = tmp % pc.h;
    tmp = tmp / pc.h;
    int c_i = tmp % pc.c;
    int n_i = tmp / pc.c;

    int k_dim = pc.c * pc.kh * pc.kw;
    float acc = 0.0;
    for (int kh_i = 0; kh_i < pc.kh; ++kh_i) {
        int h_off = ih + pc.padding - kh_i;
        if (h_off < 0 || (h_off % pc.stride) != 0) continue;
        int oh_i = h_off / pc.stride;
        if (oh_i >= pc.oh) continue;

        for (int kw_i = 0; kw_i < pc.kw; ++kw_i) {
            int w_off = iw + pc.padding - kw_i;
            if (w_off < 0 || (w_off % pc.stride) != 0) continue;
            int ow_i = w_off / pc.stride;
            if (ow_i >= pc.ow) continue;

            int row = (n_i * pc.oh + oh_i) * pc.ow + ow_i;
            int col = (c_i * pc.kh + kh_i) * pc.kw + kw_i;
            acc += dcols[row * k_dim + col];
        }
    }
    dx[idx] = acc;
}
"""

_GLSL_MAXPOOL2D_FWD = """#version 450
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer XBuf      { float x[]; };
layout(set = 0, binding = 1) writeonly buffer OutBuf   { float y[]; };
layout(set = 0, binding = 2) writeonly buffer ArgBuf   { int argmax[]; };

layout(push_constant) uniform PC {
    int n;
    int c;
    int h;
    int w;
    int k;
    int stride;
    int oh;
    int ow;
} pc;

void main() {
    int total = pc.n * pc.c * pc.oh * pc.ow;
    int idx = int(gl_GlobalInvocationID.x);
    if (idx >= total) return;

    int ow_i = idx % pc.ow;
    int tmp  = idx / pc.ow;
    int oh_i = tmp % pc.oh;
    tmp      = tmp / pc.oh;
    int c_i  = tmp % pc.c;
    int n_i  = tmp / pc.c;

    int plane = (n_i * pc.c + c_i) * pc.h * pc.w;
    int h0 = oh_i * pc.stride;
    int w0 = ow_i * pc.stride;

    float best = -3.402823466e38;
    int best_pos = h0 * pc.w + w0;
    for (int kh = 0; kh < pc.k; ++kh) {
        int ih = h0 + kh;
        for (int kw = 0; kw < pc.k; ++kw) {
            int iw = w0 + kw;
            int pos = ih * pc.w + iw;
            float v = x[plane + pos];
            if (v > best) {
                best = v;
                best_pos = pos;
            }
        }
    }
    y[idx] = best;
    argmax[idx] = best_pos;
}
"""

_GLSL_MAXPOOL2D_BWD = """#version 450
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer GOutBuf { float g_out[]; };
layout(set = 0, binding = 1) readonly buffer ArgBuf  { int argmax[]; };
layout(set = 0, binding = 2) writeonly buffer DXBuf  { float dx[]; };

layout(push_constant) uniform PC {
    int n;
    int c;
    int h;
    int w;
    int k;
    int stride;
    int oh;
    int ow;
} pc;

void main() {
    int total = pc.n * pc.c * pc.h * pc.w;
    int idx = int(gl_GlobalInvocationID.x);
    if (idx >= total) return;

    int iw  = idx % pc.w;
    int tmp = idx / pc.w;
    int ih  = tmp % pc.h;
    tmp     = tmp / pc.h;
    int c_i = tmp % pc.c;
    int n_i = tmp / pc.c;

    int pos = ih * pc.w + iw;
    int out_plane = (n_i * pc.c + c_i) * pc.oh * pc.ow;

    int oh_start = (ih < pc.k) ? 0 : ((ih - pc.k) / pc.stride + 1);
    int oh_end   = min(pc.oh - 1, ih / pc.stride);
    int ow_start = (iw < pc.k) ? 0 : ((iw - pc.k) / pc.stride + 1);
    int ow_end   = min(pc.ow - 1, iw / pc.stride);

    float acc = 0.0;
    for (int oh_i = oh_start; oh_i <= oh_end; ++oh_i) {
        for (int ow_i = ow_start; ow_i <= ow_end; ++ow_i) {
            int oidx = out_plane + oh_i * pc.ow + ow_i;
            if (argmax[oidx] == pos) {
                acc += g_out[oidx];
            }
        }
    }
    dx[idx] = acc;
}
"""


def fused_im2col(
    x: vnp.ndarray, kh: int, kw: int, stride: int, padding: int
) -> Tuple[vnp.ndarray, int, int]:
    xc = x.contiguous()
    n, c, h, w = xc.shape
    oh = (h + 2 * padding - kh) // stride + 1
    ow = (w + 2 * padding - kw) // stride + 1
    k_dim = c * kh * kw
    rows = n * oh * ow
    cols = vnp.ndarray._empty_uninit((rows, k_dim), _np.float32)
    total = rows * k_dim
    if total > 0:
        spv = _get_spv("im2col", _GLSL_IM2COL)
        pc = struct.pack("<10i", n, c, h, w, kh, kw, stride, padding, oh, ow)
        dispatch_spv_gpu(spv, [xc, cols], pc, ((total + 255) // 256, 1, 1))
    return cols, oh, ow


def fused_col2im(
    dcols: vnp.ndarray,
    x_shape: Tuple[int, int, int, int],
    kh: int,
    kw: int,
    stride: int,
    padding: int,
    oh: int,
    ow: int,
) -> vnp.ndarray:
    dc = dcols.contiguous()
    n, c, h, w = x_shape
    dx = vnp.ndarray._empty_uninit((n, c, h, w), _np.float32)
    total = n * c * h * w
    if total > 0:
        spv = _get_spv("col2im", _GLSL_COL2IM)
        pc = struct.pack("<10i", n, c, h, w, kh, kw, stride, padding, oh, ow)
        dispatch_spv_gpu(spv, [dc, dx], pc, ((total + 255) // 256, 1, 1))
    return dx


def fused_max_pool2d_fwd(
    x: vnp.ndarray, kernel_size: int = 2, stride: Optional[int] = None
) -> Tuple[vnp.ndarray, vnp.ndarray, int, int]:
    xc = x.contiguous()
    n, c, h, w = xc.shape
    k = int(kernel_size)
    st = int(stride) if stride is not None else k
    oh = (h - k) // st + 1
    ow = (w - k) // st + 1
    out = vnp.ndarray._empty_uninit((n, c, oh, ow), _np.float32)
    argmax = vnp.ndarray._empty_uninit((n, c, oh, ow), _np.int32)
    total = n * c * oh * ow
    if total > 0:
        spv = _get_spv("maxpool2d_fwd", _GLSL_MAXPOOL2D_FWD)
        pc = struct.pack("<8i", n, c, h, w, k, st, oh, ow)
        dispatch_spv_gpu(spv, [xc, out, argmax], pc, ((total + 255) // 256, 1, 1))
    return out, argmax, oh, ow


def fused_max_pool2d_bwd(
    g_out: vnp.ndarray,
    argmax: vnp.ndarray,
    x_shape: Tuple[int, int, int, int],
    kernel_size: int,
    stride: int,
    oh: int,
    ow: int,
) -> vnp.ndarray:
    gc = g_out.contiguous()
    n, c, h, w = x_shape
    dx = vnp.ndarray._empty_uninit((n, c, h, w), _np.float32)
    total = n * c * h * w
    if total > 0:
        spv = _get_spv("maxpool2d_bwd", _GLSL_MAXPOOL2D_BWD)
        pc = struct.pack("<8i", n, c, h, w, int(kernel_size), int(stride), oh, ow)
        dispatch_spv_gpu(spv, [gc, argmax, dx], pc, ((total + 255) // 256, 1, 1))
    return dx


# =====================================================================
# 8. RoPE: Single-pass SPIR-V Rotary Positional Embeddings (fwd & bwd)
# =====================================================================

_GLSL_ROPE = """#version 450
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer InBuf   { float x[]; };
layout(set = 0, binding = 1) writeonly buffer OutBuf { float y[]; };

layout(push_constant) uniform PC {
    int   bh;
    int   t_len;
    int   d;
    int   start_pos;
    float theta;
    int   inverse;
} pc;

void main() {
    int half_d = pc.d / 2;
    int total_pairs = pc.bh * pc.t_len * half_d;
    int pidx = int(gl_GlobalInvocationID.x);
    if (pidx >= total_pairs) return;

    int pair = pidx % half_d;
    int tmp  = pidx / half_d;
    int t    = tmp % pc.t_len;
    int bh_i = tmp / pc.t_len;

    float pos = float(pc.start_pos + t);
    float freq = pow(pc.theta, -float(2 * pair) / float(pc.d));
    float angle = pos * freq;
    float c = cos(angle);
    float s = sin(angle);
    if (pc.inverse != 0) s = -s;

    int base = (bh_i * pc.t_len + t) * pc.d + 2 * pair;
    float x0 = x[base];
    float x1 = x[base + 1];
    y[base]     = x0 * c - x1 * s;
    y[base + 1] = x0 * s + x1 * c;
}
"""


def fused_rope(
    x: vnp.ndarray, *, start_pos: int = 0, theta: float = 10000.0, inverse: bool = False
) -> vnp.ndarray:
    """Apply (or invert, for backward) Rotary Positional Embeddings on ``(..., T, D)`` in VRAM."""
    xc = x.contiguous()
    d = int(xc.shape[-1])
    if d % 2 != 0:
        raise ValueError(f"RoPE requires even head_dim, got {d}")
    t_len = int(xc.shape[-2])
    bh = int(xc.size) // (t_len * d) if (t_len * d) > 0 else 0
    out = vnp.ndarray._empty_uninit(xc.shape, _np.float32)
    total_pairs = bh * t_len * (d // 2)
    if total_pairs > 0:
        spv = _get_spv("rope", _GLSL_ROPE)
        pc = struct.pack("<iiiifi", bh, t_len, d, int(start_pos), float(theta), 1 if inverse else 0)
        dispatch_spv_gpu(spv, [xc, out], pc, ((total_pairs + 255) // 256, 1, 1))
    return out
