#include "vknumpy_ggml.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-vulkan.h"

#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

struct VknBuffer {
    ggml_backend_buffer_t buf;
    size_t size;
};

namespace {

std::mutex g_mutex;
ggml_backend_t g_backend = nullptr;
ggml_backend_buffer_type_t g_buft = nullptr;
ggml_gallocr_t g_galloc = nullptr;
bool g_is_vulkan = false;
std::string g_device_name = "uninitialized";
std::string g_last_error;

void set_error(const std::string& msg) {
    g_last_error = msg;
}

ggml_type to_ggml_type(int dtype) {
    switch (dtype) {
        case VKN_DTYPE_F32: return GGML_TYPE_F32;
        case VKN_DTYPE_F16: return GGML_TYPE_F16;
        case VKN_DTYPE_I32: return GGML_TYPE_I32;
        default:            return GGML_TYPE_F32;
    }
}

void bind_tensor(ggml_tensor* t, const VknBuffer* buf, size_t offset) {
    t->buffer = buf->buf;
    t->data   = static_cast<char*>(ggml_backend_buffer_get_base(buf->buf)) + offset;
    ggml_backend_buffer_init_tensor(buf->buf, t);
}

void np_shape_to_ggml_ne(const int64_t* shape, int ndim, int64_t ne[4]) {
    ne[0] = 1;
    ne[1] = 1;
    ne[2] = 1;
    ne[3] = 1;
    for (int i = 0; i < ndim; ++i) {
        ne[ndim - 1 - i] = shape[i];
    }
}

ggml_context* make_graph_ctx() {
    size_t mem_size = ggml_tensor_overhead() * 64 + ggml_graph_overhead();
    ggml_init_params params = {
        /* .mem_size   = */ mem_size,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    return ggml_init(params);
}

int run_graph(ggml_context* ctx, ggml_tensor* output_node) {
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, output_node);

    if (!ggml_gallocr_alloc_graph(g_galloc, gf)) {
        set_error("ggml_gallocr_alloc_graph failed");
        return -1;
    }

    ggml_status status = ggml_backend_graph_compute(g_backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        set_error("ggml_backend_graph_compute failed");
        return -2;
    }
    return 0;
}

int ensure_init_locked() {
    if (g_backend != nullptr) {
        return 0;
    }

    int vk_count = ggml_backend_vk_get_device_count();
    if (vk_count > 0) {
        g_backend = ggml_backend_vk_init(0);
        if (g_backend != nullptr) {
            g_is_vulkan = true;
            char desc[256] = {0};
            ggml_backend_vk_get_device_description(0, desc, sizeof(desc));
            g_device_name = desc[0] ? std::string(desc) : "Vulkan GPU 0";
        }
    }

    if (g_backend == nullptr) {
        g_backend = ggml_backend_cpu_init();
        if (g_backend == nullptr) {
            set_error("Failed to initialize both Vulkan and CPU ggml backends");
            return -1;
        }
        g_is_vulkan = false;
        g_device_name = "CPU (ggml fallback)";
    }

    g_buft = ggml_backend_get_default_buffer_type(g_backend);
    g_galloc = ggml_gallocr_new(g_buft);
    if (g_galloc == nullptr) {
        set_error("Failed to create ggml_gallocr");
        return -1;
    }
    return 0;
}

} // namespace

extern "C" {

int vkn_init(void) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return ensure_init_locked();
}

void vkn_shutdown(void) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_galloc) {
        ggml_gallocr_free(g_galloc);
        g_galloc = nullptr;
    }
    if (g_backend) {
        ggml_backend_free(g_backend);
        g_backend = nullptr;
    }
    g_buft = nullptr;
    g_is_vulkan = false;
    g_device_name = "uninitialized";
}

int vkn_is_vulkan(void) {
    std::lock_guard<std::mutex> lock(g_mutex);
    ensure_init_locked();
    return g_is_vulkan ? 1 : 0;
}

const char* vkn_device_name(void) {
    std::lock_guard<std::mutex> lock(g_mutex);
    ensure_init_locked();
    return g_device_name.c_str();
}

size_t vkn_device_free_memory(void) {
    std::lock_guard<std::mutex> lock(g_mutex);
    ensure_init_locked();
    if (!g_is_vulkan) return 0;
    size_t free_mem = 0, total_mem = 0;
    ggml_backend_vk_get_device_memory(0, &free_mem, &total_mem);
    return free_mem;
}

size_t vkn_device_total_memory(void) {
    std::lock_guard<std::mutex> lock(g_mutex);
    ensure_init_locked();
    if (!g_is_vulkan) return 0;
    size_t free_mem = 0, total_mem = 0;
    ggml_backend_vk_get_device_memory(0, &free_mem, &total_mem);
    return total_mem;
}

const char* vkn_last_error(void) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_last_error.c_str();
}

VknBuffer* vkn_buf_alloc(size_t nbytes) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (ensure_init_locked() != 0) return nullptr;

    // ggml-vulkan requires non-zero allocation size; align minimum to 64 bytes
    size_t alloc_size = nbytes > 0 ? nbytes : 64;
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(g_buft, alloc_size);
    if (!buf) {
        set_error("ggml_backend_buft_alloc_buffer failed");
        return nullptr;
    }
    auto* handle = new VknBuffer{buf, nbytes};
    return handle;
}

void vkn_buf_free(VknBuffer* buf) {
    if (!buf) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    if (buf->buf) {
        ggml_backend_buffer_free(buf->buf);
    }
    delete buf;
}

size_t vkn_buf_size(const VknBuffer* buf) {
    return buf ? buf->size : 0;
}

int vkn_buf_upload(VknBuffer* buf, size_t offset, const void* src, size_t nbytes) {
    if (!buf || (!src && nbytes > 0)) return -1;
    if (nbytes == 0) return 0;
    std::lock_guard<std::mutex> lock(g_mutex);
    if (ensure_init_locked() != 0) return -1;

    ggml_context* ctx = make_graph_ctx();
    ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, static_cast<int64_t>(nbytes));
    bind_tensor(t, buf, offset);
    ggml_backend_tensor_set(t, src, 0, nbytes);
    ggml_free(ctx);
    return 0;
}

int vkn_buf_download(const VknBuffer* buf, size_t offset, void* dst, size_t nbytes) {
    if (!buf || (!dst && nbytes > 0)) return -1;
    if (nbytes == 0) return 0;
    std::lock_guard<std::mutex> lock(g_mutex);
    if (ensure_init_locked() != 0) return -1;

    ggml_context* ctx = make_graph_ctx();
    ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, static_cast<int64_t>(nbytes));
    bind_tensor(t, buf, offset);
    ggml_backend_tensor_get(t, dst, 0, nbytes);
    ggml_free(ctx);
    return 0;
}

int vkn_buf_copy(const VknBuffer* src, size_t src_off, VknBuffer* dst, size_t dst_off, size_t nbytes) {
    if (!src || !dst) return -1;
    if (nbytes == 0) return 0;
    std::lock_guard<std::mutex> lock(g_mutex);
    if (ensure_init_locked() != 0) return -1;

    // Fast GPU-to-GPU copy when size is a multiple of 4 bytes using ggml_cpy
    if (nbytes % 4 == 0) {
        ggml_context* ctx = make_graph_ctx();
        int64_t n_f32 = static_cast<int64_t>(nbytes / 4);
        ggml_tensor* s = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_f32);
        ggml_tensor* d = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_f32);
        bind_tensor(s, src, src_off);
        bind_tensor(d, dst, dst_off);
        ggml_tensor* res = ggml_cpy(ctx, s, d);
        int rc = run_graph(ctx, res);
        ggml_free(ctx);
        return rc;
    }

    // General byte-level fallback via ggml_backend_tensor_copy
    ggml_context* ctx = make_graph_ctx();
    ggml_tensor* s = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, static_cast<int64_t>(nbytes));
    ggml_tensor* d = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, static_cast<int64_t>(nbytes));
    bind_tensor(s, src, src_off);
    bind_tensor(d, dst, dst_off);
    ggml_backend_tensor_copy(s, d);
    ggml_free(ctx);
    return 0;
}

int vkn_matmul(
    const VknBuffer* a, size_t a_off, const int64_t* a_shape, int a_ndim,
    const VknBuffer* b, size_t b_off, const int64_t* b_shape, int b_ndim,
    int b_transposed,
    VknBuffer* out, size_t out_off,
    int dtype
) {
    if (!a || !b || !out || a_ndim < 2 || a_ndim > 4 || b_ndim < 2 || b_ndim > 4) {
        set_error("vkn_matmul: invalid arguments or ndim (must be 2..4)");
        return -1;
    }

    std::lock_guard<std::mutex> lock(g_mutex);
    if (ensure_init_locked() != 0) return -1;

    ggml_type gtype = to_ggml_type(dtype);
    int64_t a_ne[4], b_ne[4];
    np_shape_to_ggml_ne(a_shape, a_ndim, a_ne);
    np_shape_to_ggml_ne(b_shape, b_ndim, b_ne);

    // a_shape is (..., M, K) -> a_ne = [K, M, a_b2, a_b3]
    // If b_transposed == 0: b_shape is (..., K, N) -> b_ne = [N, K, b_b2, b_b3]
    // If b_transposed == 1: b_shape is (..., N, K) -> b_ne = [K, N, b_b2, b_b3]
    int64_t K = a_ne[0];
    int64_t M = a_ne[1];
    int64_t N = b_transposed ? b_ne[1] : b_ne[0];
    int64_t b_K = b_transposed ? b_ne[0] : b_ne[1];

    if (K != b_K) {
        set_error("vkn_matmul: inner dimension mismatch");
        return -1;
    }

    int64_t out_b2 = (a_ne[2] > b_ne[2]) ? a_ne[2] : b_ne[2];
    int64_t out_b3 = (a_ne[3] > b_ne[3]) ? a_ne[3] : b_ne[3];

    if ((a_ne[2] != out_b2 && a_ne[2] != 1) || (b_ne[2] != out_b2 && b_ne[2] != 1) ||
        (a_ne[3] != out_b3 && a_ne[3] != 1) || (b_ne[3] != out_b3 && b_ne[3] != 1)) {
        set_error("vkn_matmul: incompatible batch dimensions");
        return -1;
    }

    ggml_context* ctx = make_graph_ctx();

    ggml_tensor* ta = ggml_new_tensor_4d(ctx, gtype, a_ne[0], a_ne[1], a_ne[2], a_ne[3]);
    ggml_tensor* tb = ggml_new_tensor_4d(ctx, gtype, b_ne[0], b_ne[1], b_ne[2], b_ne[3]);
    bind_tensor(ta, a, a_off);
    bind_tensor(tb, b, b_off);

    // ggml_mul_mat(ctx, src0=tb_T, src1=ta) requires:
    //   src0: [K, N, b_b2, b_b3]
    //   src1: [K, M, out_b2, out_b3]  (src1->ne[2] % src0->ne[2] == 0)
    //   result: [N, M, out_b2, out_b3] (F32)
    ggml_tensor* src0 = tb;
    if (!b_transposed) {
        src0 = ggml_cont(ctx, ggml_transpose(ctx, tb));
    }

    ggml_tensor* src1 = ta;
    if (a_ne[2] != out_b2 || a_ne[3] != out_b3) {
        ggml_tensor* tmpl = ggml_new_tensor_4d(ctx, gtype, K, M, out_b2, out_b3);
        src1 = ggml_repeat(ctx, ta, tmpl);
    }

    // ggml_mul_mat expects src1 to be F32 when src0 is F16 (or both F32)
    if (src1->type != GGML_TYPE_F32) {
        ggml_tensor* src1_f32 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]);
        src1 = ggml_cpy(ctx, src1, src1_f32);
    }

    ggml_tensor* mm = ggml_mul_mat(ctx, src0, src1);
    if (gtype == GGML_TYPE_F32) {
        ggml_prec_set_acc(mm, GGML_PREC_F32);
        ggml_prec_set_src(mm, GGML_PREC_F32, 0);
        ggml_prec_set_src(mm, GGML_PREC_F32, 1);
    }
    ggml_tensor* final_node = mm;

    if (gtype == GGML_TYPE_F32) {
        bind_tensor(mm, out, out_off);
    } else {
        ggml_tensor* out_t = ggml_new_tensor_4d(ctx, gtype, N, M, out_b2, out_b3);
        bind_tensor(out_t, out, out_off);
        final_node = ggml_cpy(ctx, mm, out_t);
    }

    int rc = run_graph(ctx, final_node);
    ggml_free(ctx);
    return rc;
}

int vkn_dot_1d(
    const VknBuffer* a, size_t a_off,
    const VknBuffer* b, size_t b_off,
    int64_t n,
    VknBuffer* out, size_t out_off,
    int dtype
) {
    int64_t shape[2] = {1, n};
    // a is (1, n), b is (1, n) treated as b_transposed=1 -> (1, 1)
    return vkn_matmul(a, a_off, shape, 2, b, b_off, shape, 2, 1, out, out_off, dtype);
}

int vkn_outer(
    const VknBuffer* a, size_t a_off, int64_t m,
    const VknBuffer* b, size_t b_off, int64_t n,
    VknBuffer* out, size_t out_off,
    int dtype
) {
    int64_t a_shape[2] = {m, 1};
    int64_t b_shape[2] = {n, 1};
    // a is (m, 1), b is (n, 1) with b_transposed=1 -> a @ b.T = (m, n)
    return vkn_matmul(a, a_off, a_shape, 2, b, b_off, b_shape, 2, 1, out, out_off, dtype);
}

int vkn_binary_op(
    int op,
    const VknBuffer* a, size_t a_off, const int64_t* a_shape,
    const VknBuffer* b, size_t b_off, const int64_t* b_shape,
    VknBuffer* out, size_t out_off, const int64_t* out_shape,
    int ndim,
    int dtype
) {
    if (!a || !b || !out || ndim < 1 || ndim > 4) {
        set_error("vkn_binary_op: invalid arguments or ndim (must be 1..4)");
        return -1;
    }

    std::lock_guard<std::mutex> lock(g_mutex);
    if (ensure_init_locked() != 0) return -1;

    ggml_type gtype = to_ggml_type(dtype);
    int64_t a_ne[4], b_ne[4], o_ne[4];
    np_shape_to_ggml_ne(a_shape, ndim, a_ne);
    np_shape_to_ggml_ne(b_shape, ndim, b_ne);
    np_shape_to_ggml_ne(out_shape, ndim, o_ne);

    ggml_context* ctx = make_graph_ctx();

    ggml_tensor* ta = ggml_new_tensor_4d(ctx, gtype, a_ne[0], a_ne[1], a_ne[2], a_ne[3]);
    ggml_tensor* tb = ggml_new_tensor_4d(ctx, gtype, b_ne[0], b_ne[1], b_ne[2], b_ne[3]);
    bind_tensor(ta, a, a_off);
    bind_tensor(tb, b, b_off);

    // Promote F16 -> F32 for full Vulkan op coverage, then cast back
    ggml_tensor* sa = ta;
    ggml_tensor* sb = tb;
    if (gtype != GGML_TYPE_F32) {
        ggml_tensor* sa_f32 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, a_ne[0], a_ne[1], a_ne[2], a_ne[3]);
        ggml_tensor* sb_f32 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, b_ne[0], b_ne[1], b_ne[2], b_ne[3]);
        sa = ggml_cpy(ctx, ta, sa_f32);
        sb = ggml_cpy(ctx, tb, sb_f32);
    }

    // ggml binary ops require src0 to match output shape
    bool a_needs_repeat = (a_ne[0] != o_ne[0] || a_ne[1] != o_ne[1] || a_ne[2] != o_ne[2] || a_ne[3] != o_ne[3]);
    if (a_needs_repeat) {
        ggml_tensor* tmpl = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, o_ne[0], o_ne[1], o_ne[2], o_ne[3]);
        sa = ggml_repeat(ctx, sa, tmpl);
    }

    ggml_tensor* res = nullptr;
    switch (op) {
        case VKN_BIN_ADD: res = ggml_add(ctx, sa, sb); break;
        case VKN_BIN_SUB: res = ggml_sub(ctx, sa, sb); break;
        case VKN_BIN_MUL: res = ggml_mul(ctx, sa, sb); break;
        case VKN_BIN_DIV: res = ggml_div(ctx, sa, sb); break;
        case VKN_BIN_MAX: {
            ggml_tensor* sum  = ggml_add(ctx, sa, sb);
            ggml_tensor* adif = ggml_abs(ctx, ggml_sub(ctx, sa, sb));
            res = ggml_scale(ctx, ggml_add(ctx, sum, adif), 0.5f);
            break;
        }
        case VKN_BIN_MIN: {
            ggml_tensor* sum  = ggml_add(ctx, sa, sb);
            ggml_tensor* adif = ggml_abs(ctx, ggml_sub(ctx, sa, sb));
            res = ggml_scale(ctx, ggml_sub(ctx, sum, adif), 0.5f);
            break;
        }
        case VKN_BIN_GT: {
            ggml_tensor* diff = ggml_sub(ctx, sa, sb);
            res = ggml_step(ctx, diff);
            break;
        }
        case VKN_BIN_LT: {
            ggml_tensor* ndiff = ggml_neg(ctx, ggml_sub(ctx, sa, sb));
            res = ggml_step(ctx, ndiff);
            break;
        }
        case VKN_BIN_GE: {
            ggml_tensor* ndiff = ggml_neg(ctx, ggml_sub(ctx, sa, sb));
            res = ggml_scale_bias(ctx, ggml_step(ctx, ndiff), -1.0f, 1.0f);
            break;
        }
        case VKN_BIN_LE: {
            ggml_tensor* diff = ggml_sub(ctx, sa, sb);
            res = ggml_scale_bias(ctx, ggml_step(ctx, diff), -1.0f, 1.0f);
            break;
        }
        case VKN_BIN_EQ: {
            ggml_tensor* adif = ggml_abs(ctx, ggml_sub(ctx, sa, sb));
            res = ggml_scale_bias(ctx, ggml_step(ctx, adif), -1.0f, 1.0f);
            break;
        }
        case VKN_BIN_NE: {
            ggml_tensor* adif = ggml_abs(ctx, ggml_sub(ctx, sa, sb));
            res = ggml_step(ctx, adif);
            break;
        }
        default:
            ggml_free(ctx);
            set_error("vkn_binary_op: unknown op");
            return -1;
    }

    ggml_tensor* final_node = res;
    if (gtype == GGML_TYPE_F32) {
        bind_tensor(res, out, out_off);
    } else {
        ggml_tensor* out_t = ggml_new_tensor_4d(ctx, gtype, o_ne[0], o_ne[1], o_ne[2], o_ne[3]);
        bind_tensor(out_t, out, out_off);
        final_node = ggml_cpy(ctx, res, out_t);
    }

    int rc = run_graph(ctx, final_node);
    ggml_free(ctx);
    return rc;
}

int vkn_unary_op(
    int op,
    const VknBuffer* a, size_t a_off,
    int64_t n,
    VknBuffer* out, size_t out_off,
    int dtype
) {
    if (!a || !out || n <= 0) {
        set_error("vkn_unary_op: invalid arguments");
        return -1;
    }

    std::lock_guard<std::mutex> lock(g_mutex);
    if (ensure_init_locked() != 0) return -1;

    ggml_type gtype = to_ggml_type(dtype);
    ggml_context* ctx = make_graph_ctx();

    ggml_tensor* ta = ggml_new_tensor_1d(ctx, gtype, n);
    bind_tensor(ta, a, a_off);

    ggml_tensor* sa = ta;
    if (gtype != GGML_TYPE_F32) {
        ggml_tensor* sa_f32 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        sa = ggml_cpy(ctx, ta, sa_f32);
    }

    ggml_tensor* res = nullptr;
    switch (op) {
        case VKN_UNARY_NEG:     res = ggml_neg(ctx, sa); break;
        case VKN_UNARY_ABS:     res = ggml_abs(ctx, sa); break;
        case VKN_UNARY_SQR:     res = ggml_sqr(ctx, sa); break;
        case VKN_UNARY_SQRT:    res = ggml_sqrt(ctx, sa); break;
        case VKN_UNARY_EXP:     res = ggml_exp(ctx, sa); break;
        case VKN_UNARY_LOG:     res = ggml_log(ctx, sa); break;
        case VKN_UNARY_SIN:     res = ggml_sin(ctx, sa); break;
        case VKN_UNARY_COS:     res = ggml_cos(ctx, sa); break;
        case VKN_UNARY_TANH:    res = ggml_tanh(ctx, sa); break;
        case VKN_UNARY_RELU:    res = ggml_relu(ctx, sa); break;
        case VKN_UNARY_SIGMOID: res = ggml_sigmoid(ctx, sa); break;
        case VKN_UNARY_SILU:    res = ggml_silu(ctx, sa); break;
        case VKN_UNARY_GELU:    res = ggml_gelu(ctx, sa); break;
        case VKN_UNARY_FLOOR:   res = ggml_floor(ctx, sa); break;
        case VKN_UNARY_CEIL:    res = ggml_ceil(ctx, sa); break;
        case VKN_UNARY_ROUND:   res = ggml_round(ctx, sa); break;
        case VKN_UNARY_TRUNC:   res = ggml_trunc(ctx, sa); break;
        case VKN_UNARY_SGN:     res = ggml_sgn(ctx, sa); break;
        case VKN_UNARY_EXPM1:   res = ggml_expm1(ctx, sa); break;
        case VKN_UNARY_STEP:    res = ggml_step(ctx, sa); break;
        default:
            ggml_free(ctx);
            set_error("vkn_unary_op: unknown op");
            return -1;
    }

    ggml_tensor* final_node = res;
    if (gtype == GGML_TYPE_F32) {
        bind_tensor(res, out, out_off);
    } else {
        ggml_tensor* out_t = ggml_new_tensor_1d(ctx, gtype, n);
        bind_tensor(out_t, out, out_off);
        final_node = ggml_cpy(ctx, res, out_t);
    }

    int rc = run_graph(ctx, final_node);
    ggml_free(ctx);
    return rc;
}

int vkn_scale_bias(
    const VknBuffer* a, size_t a_off,
    int64_t n,
    float scale,
    float bias,
    VknBuffer* out, size_t out_off,
    int dtype
) {
    if (!a || !out || n <= 0) return -1;
    std::lock_guard<std::mutex> lock(g_mutex);
    if (ensure_init_locked() != 0) return -1;

    ggml_type gtype = to_ggml_type(dtype);
    ggml_context* ctx = make_graph_ctx();

    ggml_tensor* ta = ggml_new_tensor_1d(ctx, gtype, n);
    bind_tensor(ta, a, a_off);

    ggml_tensor* sa = ta;
    if (gtype != GGML_TYPE_F32) {
        ggml_tensor* sa_f32 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        sa = ggml_cpy(ctx, ta, sa_f32);
    }

    ggml_tensor* res = ggml_scale_bias(ctx, sa, scale, bias);
    ggml_tensor* final_node = res;

    if (gtype == GGML_TYPE_F32) {
        bind_tensor(res, out, out_off);
    } else {
        ggml_tensor* out_t = ggml_new_tensor_1d(ctx, gtype, n);
        bind_tensor(out_t, out, out_off);
        final_node = ggml_cpy(ctx, res, out_t);
    }

    int rc = run_graph(ctx, final_node);
    ggml_free(ctx);
    return rc;
}

int vkn_clamp(
    const VknBuffer* a, size_t a_off,
    int64_t n,
    float min_val,
    float max_val,
    VknBuffer* out, size_t out_off,
    int dtype
) {
    if (!a || !out || n <= 0) return -1;
    std::lock_guard<std::mutex> lock(g_mutex);
    if (ensure_init_locked() != 0) return -1;

    ggml_type gtype = to_ggml_type(dtype);
    ggml_context* ctx = make_graph_ctx();

    ggml_tensor* ta = ggml_new_tensor_1d(ctx, gtype, n);
    bind_tensor(ta, a, a_off);

    ggml_tensor* sa = ta;
    if (gtype != GGML_TYPE_F32) {
        ggml_tensor* sa_f32 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        sa = ggml_cpy(ctx, ta, sa_f32);
    }

    ggml_tensor* res = ggml_clamp(ctx, sa, min_val, max_val);
    ggml_tensor* final_node = res;

    if (gtype == GGML_TYPE_F32) {
        bind_tensor(res, out, out_off);
    } else {
        ggml_tensor* out_t = ggml_new_tensor_1d(ctx, gtype, n);
        bind_tensor(out_t, out, out_off);
        final_node = ggml_cpy(ctx, res, out_t);
    }

    int rc = run_graph(ctx, final_node);
    ggml_free(ctx);
    return rc;
}

int vkn_sum_all(
    const VknBuffer* a, size_t a_off,
    int64_t n,
    VknBuffer* out, size_t out_off,
    int dtype
) {
    if (!a || !out || n <= 0) return -1;
    std::lock_guard<std::mutex> lock(g_mutex);
    if (ensure_init_locked() != 0) return -1;

    ggml_type gtype = to_ggml_type(dtype);
    ggml_context* ctx = make_graph_ctx();

    ggml_tensor* ta = ggml_new_tensor_1d(ctx, gtype, n);
    bind_tensor(ta, a, a_off);

    ggml_tensor* sa = ta;
    if (gtype != GGML_TYPE_F32) {
        ggml_tensor* sa_f32 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        sa = ggml_cpy(ctx, ta, sa_f32);
    }

    ggml_tensor* res = ggml_sum(ctx, sa);
    ggml_tensor* final_node = res;

    if (gtype == GGML_TYPE_F32) {
        bind_tensor(res, out, out_off);
    } else {
        ggml_tensor* out_t = ggml_new_tensor_1d(ctx, gtype, 1);
        bind_tensor(out_t, out, out_off);
        final_node = ggml_cpy(ctx, res, out_t);
    }

    int rc = run_graph(ctx, final_node);
    ggml_free(ctx);
    return rc;
}

int vkn_sum_rows(
    const VknBuffer* a, size_t a_off,
    int64_t rows, int64_t cols,
    VknBuffer* out, size_t out_off,
    int dtype
) {
    if (!a || !out || rows <= 0 || cols <= 0) return -1;
    std::lock_guard<std::mutex> lock(g_mutex);
    if (ensure_init_locked() != 0) return -1;

    ggml_type gtype = to_ggml_type(dtype);
    ggml_context* ctx = make_graph_ctx();

    // NumPy (rows, cols) -> ggml [cols, rows]
    ggml_tensor* ta = ggml_new_tensor_2d(ctx, gtype, cols, rows);
    bind_tensor(ta, a, a_off);

    ggml_tensor* sa = ta;
    if (gtype != GGML_TYPE_F32) {
        ggml_tensor* sa_f32 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);
        sa = ggml_cpy(ctx, ta, sa_f32);
    }

    ggml_tensor* res = ggml_sum_rows(ctx, sa); // [1, rows]
    ggml_tensor* final_node = res;

    if (gtype == GGML_TYPE_F32) {
        bind_tensor(res, out, out_off);
    } else {
        ggml_tensor* out_t = ggml_new_tensor_2d(ctx, gtype, 1, rows);
        bind_tensor(out_t, out, out_off);
        final_node = ggml_cpy(ctx, res, out_t);
    }

    int rc = run_graph(ctx, final_node);
    ggml_free(ctx);
    return rc;
}

int vkn_mean_rows(
    const VknBuffer* a, size_t a_off,
    int64_t rows, int64_t cols,
    VknBuffer* out, size_t out_off,
    int dtype
) {
    if (!a || !out || rows <= 0 || cols <= 0) return -1;
    std::lock_guard<std::mutex> lock(g_mutex);
    if (ensure_init_locked() != 0) return -1;

    ggml_type gtype = to_ggml_type(dtype);
    ggml_context* ctx = make_graph_ctx();

    ggml_tensor* ta = ggml_new_tensor_2d(ctx, gtype, cols, rows);
    bind_tensor(ta, a, a_off);

    ggml_tensor* sa = ta;
    if (gtype != GGML_TYPE_F32) {
        ggml_tensor* sa_f32 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);
        sa = ggml_cpy(ctx, ta, sa_f32);
    }

    ggml_tensor* res = ggml_mean(ctx, sa); // [1, rows]
    ggml_tensor* final_node = res;

    if (gtype == GGML_TYPE_F32) {
        bind_tensor(res, out, out_off);
    } else {
        ggml_tensor* out_t = ggml_new_tensor_2d(ctx, gtype, 1, rows);
        bind_tensor(out_t, out, out_off);
        final_node = ggml_cpy(ctx, res, out_t);
    }

    int rc = run_graph(ctx, final_node);
    ggml_free(ctx);
    return rc;
}

int vkn_argmax_rows(
    const VknBuffer* a, size_t a_off,
    int64_t rows, int64_t cols,
    VknBuffer* out_i32, size_t out_off,
    int dtype
) {
    if (!a || !out_i32 || rows <= 0 || cols <= 0) return -1;
    std::lock_guard<std::mutex> lock(g_mutex);
    if (ensure_init_locked() != 0) return -1;

    ggml_type gtype = to_ggml_type(dtype);
    ggml_context* ctx = make_graph_ctx();

    ggml_tensor* ta = ggml_new_tensor_2d(ctx, gtype, cols, rows);
    bind_tensor(ta, a, a_off);

    ggml_tensor* sa = ta;
    if (gtype != GGML_TYPE_F32) {
        ggml_tensor* sa_f32 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);
        sa = ggml_cpy(ctx, ta, sa_f32);
    }

    ggml_tensor* res = ggml_argmax(ctx, sa); // [rows] I32
    bind_tensor(res, out_i32, out_off);

    int rc = run_graph(ctx, res);
    ggml_free(ctx);
    return rc;
}

int vkn_transpose_cont(
    const VknBuffer* a, size_t a_off,
    const int64_t* shape, const int* perm, int ndim,
    VknBuffer* out, size_t out_off,
    int dtype
) {
    if (!a || !out || ndim < 2 || ndim > 4) return -1;
    std::lock_guard<std::mutex> lock(g_mutex);
    if (ensure_init_locked() != 0) return -1;

    ggml_type gtype = to_ggml_type(dtype);
    int64_t ne[4];
    np_shape_to_ggml_ne(shape, ndim, ne);

    int axis[4] = {0, 1, 2, 3};
    for (int np_dst = 0; np_dst < ndim; ++np_dst) {
        int np_src = perm[np_dst];
        int ggml_src = ndim - 1 - np_src;
        int ggml_dst = ndim - 1 - np_dst;
        axis[ggml_src] = ggml_dst;
    }

    ggml_context* ctx = make_graph_ctx();
    ggml_tensor* ta = ggml_new_tensor_4d(ctx, gtype, ne[0], ne[1], ne[2], ne[3]);
    bind_tensor(ta, a, a_off);

    ggml_tensor* permuted = ggml_permute(ctx, ta, axis[0], axis[1], axis[2], axis[3]);
    ggml_tensor* res = ggml_cont(ctx, permuted);
    bind_tensor(res, out, out_off);

    int rc = run_graph(ctx, res);
    ggml_free(ctx);
    return rc;
}

int vkn_cast(
    const VknBuffer* a, size_t a_off, int src_dtype,
    int64_t n,
    VknBuffer* out, size_t out_off, int dst_dtype
) {
    if (!a || !out || n <= 0) return -1;
    std::lock_guard<std::mutex> lock(g_mutex);
    if (ensure_init_locked() != 0) return -1;

    ggml_context* ctx = make_graph_ctx();
    ggml_tensor* ta = ggml_new_tensor_1d(ctx, to_ggml_type(src_dtype), n);
    ggml_tensor* tb = ggml_new_tensor_1d(ctx, to_ggml_type(dst_dtype), n);
    bind_tensor(ta, a, a_off);
    bind_tensor(tb, out, out_off);

    ggml_tensor* res = ggml_cpy(ctx, ta, tb);
    int rc = run_graph(ctx, res);
    ggml_free(ctx);
    return rc;
}

/* ======================================================================
 * Phase 1: Native ggml-vulkan creation, manipulation, sorting, gather & solve_tri
 * ====================================================================== */

int vkn_fill(
    VknBuffer* out, size_t out_off,
    int64_t n,
    float value,
    int dtype
) {
    if (!out || n <= 0) return -1;
    std::lock_guard<std::mutex> lock(g_mutex);
    if (ensure_init_locked() != 0) return -1;

    ggml_type gtype = to_ggml_type(dtype);
    ggml_context* ctx = make_graph_ctx();

    ggml_tensor* final_node = nullptr;
    if (gtype == GGML_TYPE_F32) {
        ggml_tensor* dst = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        bind_tensor(dst, out, out_off);
        final_node = ggml_fill_inplace(ctx, dst, value);
    } else {
        ggml_tensor* tmp = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        ggml_tensor* filled = ggml_fill_inplace(ctx, tmp, value);
        ggml_tensor* dst = ggml_new_tensor_1d(ctx, gtype, n);
        bind_tensor(dst, out, out_off);
        final_node = ggml_cpy(ctx, filled, dst);
    }

    int rc = run_graph(ctx, final_node);
    ggml_free(ctx);
    return rc;
}

int vkn_arange(
    float start, float stop, float step,
    VknBuffer* out, size_t out_off,
    int dtype
) {
    if (!out || step == 0.0f) return -1;
    std::lock_guard<std::mutex> lock(g_mutex);
    if (ensure_init_locked() != 0) return -1;

    ggml_type gtype = to_ggml_type(dtype);
    ggml_context* ctx = make_graph_ctx();

    ggml_tensor* res = ggml_arange(ctx, start, stop, step);
    ggml_tensor* final_node = res;

    if (gtype == GGML_TYPE_F32) {
        bind_tensor(res, out, out_off);
    } else {
        ggml_tensor* dst = ggml_new_tensor_1d(ctx, gtype, res->ne[0]);
        bind_tensor(dst, out, out_off);
        final_node = ggml_cpy(ctx, res, dst);
    }

    int rc = run_graph(ctx, final_node);
    ggml_free(ctx);
    return rc;
}

int vkn_diag(
    const VknBuffer* a, size_t a_off,
    int64_t n,
    VknBuffer* out, size_t out_off,
    int dtype
) {
    if (!a || !out || n <= 0) return -1;
    std::lock_guard<std::mutex> lock(g_mutex);
    if (ensure_init_locked() != 0) return -1;

    ggml_type gtype = to_ggml_type(dtype);
    ggml_context* ctx = make_graph_ctx();

    ggml_tensor* ta = ggml_new_tensor_1d(ctx, gtype, n);
    bind_tensor(ta, a, a_off);

    ggml_tensor* sa = ta;
    if (gtype != GGML_TYPE_F32) {
        ggml_tensor* sa_f32 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        sa = ggml_cpy(ctx, ta, sa_f32);
    }

    ggml_tensor* res = ggml_diag(ctx, sa);
    ggml_tensor* final_node = res;

    if (gtype == GGML_TYPE_F32) {
        bind_tensor(res, out, out_off);
    } else {
        ggml_tensor* dst = ggml_new_tensor_2d(ctx, gtype, n, n);
        bind_tensor(dst, out, out_off);
        final_node = ggml_cpy(ctx, res, dst);
    }

    int rc = run_graph(ctx, final_node);
    ggml_free(ctx);
    return rc;
}

int vkn_tri(
    const VknBuffer* a, size_t a_off,
    const int64_t* shape, int ndim,
    int tri_type,
    VknBuffer* out, size_t out_off,
    int dtype
) {
    if (!a || !out || ndim < 2 || ndim > 4) return -1;
    std::lock_guard<std::mutex> lock(g_mutex);
    if (ensure_init_locked() != 0) return -1;

    ggml_type gtype = to_ggml_type(dtype);
    int64_t ne[4];
    np_shape_to_ggml_ne(shape, ndim, ne);

    ggml_context* ctx = make_graph_ctx();
    ggml_tensor* ta = ggml_new_tensor_4d(ctx, gtype, ne[0], ne[1], ne[2], ne[3]);
    bind_tensor(ta, a, a_off);

    ggml_tensor* sa = ta;
    if (gtype != GGML_TYPE_F32) {
        ggml_tensor* sa_f32 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]);
        sa = ggml_cpy(ctx, ta, sa_f32);
    }

    ggml_tensor* res = ggml_tri(ctx, sa, static_cast<ggml_tri_type>(tri_type));
    ggml_tensor* final_node = res;

    if (gtype == GGML_TYPE_F32) {
        bind_tensor(res, out, out_off);
    } else {
        ggml_tensor* dst = ggml_new_tensor_4d(ctx, gtype, ne[0], ne[1], ne[2], ne[3]);
        bind_tensor(dst, out, out_off);
        final_node = ggml_cpy(ctx, res, dst);
    }

    int rc = run_graph(ctx, final_node);
    ggml_free(ctx);
    return rc;
}

int vkn_concat(
    const VknBuffer* a, size_t a_off, const int64_t* a_shape,
    const VknBuffer* b, size_t b_off, const int64_t* b_shape,
    int ndim, int axis,
    VknBuffer* out, size_t out_off,
    int dtype
) {
    if (!a || !b || !out || ndim < 1 || ndim > 4 || axis < 0 || axis >= ndim) return -1;
    std::lock_guard<std::mutex> lock(g_mutex);
    if (ensure_init_locked() != 0) return -1;

    ggml_type gtype = to_ggml_type(dtype);
    int64_t a_ne[4], b_ne[4];
    np_shape_to_ggml_ne(a_shape, ndim, a_ne);
    np_shape_to_ggml_ne(b_shape, ndim, b_ne);
    int ggml_dim = ndim - 1 - axis;

    ggml_context* ctx = make_graph_ctx();
    ggml_tensor* ta = ggml_new_tensor_4d(ctx, gtype, a_ne[0], a_ne[1], a_ne[2], a_ne[3]);
    ggml_tensor* tb = ggml_new_tensor_4d(ctx, gtype, b_ne[0], b_ne[1], b_ne[2], b_ne[3]);
    bind_tensor(ta, a, a_off);
    bind_tensor(tb, b, b_off);

    ggml_tensor* res = ggml_concat(ctx, ta, tb, ggml_dim);
    bind_tensor(res, out, out_off);

    int rc = run_graph(ctx, res);
    ggml_free(ctx);
    return rc;
}

int vkn_pad(
    const VknBuffer* a, size_t a_off,
    const int64_t* shape, int ndim,
    const int* pad_before, const int* pad_after,
    VknBuffer* out, size_t out_off,
    int dtype
) {
    if (!a || !out || ndim < 1 || ndim > 4) return -1;
    std::lock_guard<std::mutex> lock(g_mutex);
    if (ensure_init_locked() != 0) return -1;

    ggml_type gtype = to_ggml_type(dtype);
    int64_t ne[4];
    np_shape_to_ggml_ne(shape, ndim, ne);

    int lp[4] = {0, 0, 0, 0};
    int rp[4] = {0, 0, 0, 0};
    for (int i = 0; i < ndim; ++i) {
        lp[ndim - 1 - i] = pad_before[i];
        rp[ndim - 1 - i] = pad_after[i];
    }

    ggml_context* ctx = make_graph_ctx();
    ggml_tensor* ta = ggml_new_tensor_4d(ctx, gtype, ne[0], ne[1], ne[2], ne[3]);
    bind_tensor(ta, a, a_off);

    ggml_tensor* sa = ta;
    if (gtype != GGML_TYPE_F32) {
        ggml_tensor* sa_f32 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]);
        sa = ggml_cpy(ctx, ta, sa_f32);
    }

    ggml_tensor* res = ggml_pad_ext(ctx, sa, lp[0], rp[0], lp[1], rp[1], lp[2], rp[2], lp[3], rp[3]);
    ggml_tensor* final_node = res;

    if (gtype == GGML_TYPE_F32) {
        bind_tensor(res, out, out_off);
    } else {
        ggml_tensor* dst = ggml_new_tensor_4d(ctx, gtype, res->ne[0], res->ne[1], res->ne[2], res->ne[3]);
        bind_tensor(dst, out, out_off);
        final_node = ggml_cpy(ctx, res, dst);
    }

    int rc = run_graph(ctx, final_node);
    ggml_free(ctx);
    return rc;
}

int vkn_roll(
    const VknBuffer* a, size_t a_off,
    const int64_t* shape, int ndim,
    const int* shifts,
    VknBuffer* out, size_t out_off,
    int dtype
) {
    if (!a || !out || ndim < 1 || ndim > 4) return -1;
    std::lock_guard<std::mutex> lock(g_mutex);
    if (ensure_init_locked() != 0) return -1;

    ggml_type gtype = to_ggml_type(dtype);
    int64_t ne[4];
    np_shape_to_ggml_ne(shape, ndim, ne);

    int sh[4] = {0, 0, 0, 0};
    for (int i = 0; i < ndim; ++i) {
        sh[ndim - 1 - i] = shifts[i];
    }

    ggml_context* ctx = make_graph_ctx();
    ggml_tensor* ta = ggml_new_tensor_4d(ctx, gtype, ne[0], ne[1], ne[2], ne[3]);
    bind_tensor(ta, a, a_off);

    ggml_tensor* sa = ta;
    if (gtype != GGML_TYPE_F32) {
        ggml_tensor* sa_f32 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]);
        sa = ggml_cpy(ctx, ta, sa_f32);
    }

    ggml_tensor* res = ggml_roll(ctx, sa, sh[0], sh[1], sh[2], sh[3]);
    ggml_tensor* final_node = res;

    if (gtype == GGML_TYPE_F32) {
        bind_tensor(res, out, out_off);
    } else {
        ggml_tensor* dst = ggml_new_tensor_4d(ctx, gtype, ne[0], ne[1], ne[2], ne[3]);
        bind_tensor(dst, out, out_off);
        final_node = ggml_cpy(ctx, res, dst);
    }

    int rc = run_graph(ctx, final_node);
    ggml_free(ctx);
    return rc;
}

int vkn_repeat(
    const VknBuffer* a, size_t a_off, const int64_t* a_shape,
    VknBuffer* out, size_t out_off, const int64_t* out_shape,
    int ndim,
    int dtype
) {
    if (!a || !out || ndim < 1 || ndim > 4) return -1;
    std::lock_guard<std::mutex> lock(g_mutex);
    if (ensure_init_locked() != 0) return -1;

    ggml_type gtype = to_ggml_type(dtype);
    int64_t a_ne[4], o_ne[4];
    np_shape_to_ggml_ne(a_shape, ndim, a_ne);
    np_shape_to_ggml_ne(out_shape, ndim, o_ne);

    ggml_context* ctx = make_graph_ctx();
    ggml_tensor* ta = ggml_new_tensor_4d(ctx, gtype, a_ne[0], a_ne[1], a_ne[2], a_ne[3]);
    bind_tensor(ta, a, a_off);

    ggml_tensor* sa = ta;
    if (gtype != GGML_TYPE_F32) {
        ggml_tensor* sa_f32 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, a_ne[0], a_ne[1], a_ne[2], a_ne[3]);
        sa = ggml_cpy(ctx, ta, sa_f32);
    }

    ggml_tensor* tmpl = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, o_ne[0], o_ne[1], o_ne[2], o_ne[3]);
    ggml_tensor* res = ggml_repeat(ctx, sa, tmpl);
    ggml_tensor* final_node = res;

    if (gtype == GGML_TYPE_F32) {
        bind_tensor(res, out, out_off);
    } else {
        ggml_tensor* dst = ggml_new_tensor_4d(ctx, gtype, o_ne[0], o_ne[1], o_ne[2], o_ne[3]);
        bind_tensor(dst, out, out_off);
        final_node = ggml_cpy(ctx, res, dst);
    }

    int rc = run_graph(ctx, final_node);
    ggml_free(ctx);
    return rc;
}

int vkn_cumsum_rows(
    const VknBuffer* a, size_t a_off,
    int64_t rows, int64_t cols,
    VknBuffer* out, size_t out_off,
    int dtype
) {
    if (!a || !out || rows <= 0 || cols <= 0) return -1;
    std::lock_guard<std::mutex> lock(g_mutex);
    if (ensure_init_locked() != 0) return -1;

    ggml_type gtype = to_ggml_type(dtype);
    ggml_context* ctx = make_graph_ctx();

    ggml_tensor* ta = ggml_new_tensor_2d(ctx, gtype, cols, rows);
    bind_tensor(ta, a, a_off);

    ggml_tensor* sa = ta;
    if (gtype != GGML_TYPE_F32) {
        ggml_tensor* sa_f32 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);
        sa = ggml_cpy(ctx, ta, sa_f32);
    }

    ggml_tensor* res = ggml_cumsum(ctx, sa);
    ggml_tensor* final_node = res;

    if (gtype == GGML_TYPE_F32) {
        bind_tensor(res, out, out_off);
    } else {
        ggml_tensor* dst = ggml_new_tensor_2d(ctx, gtype, cols, rows);
        bind_tensor(dst, out, out_off);
        final_node = ggml_cpy(ctx, res, dst);
    }

    int rc = run_graph(ctx, final_node);
    ggml_free(ctx);
    return rc;
}

int vkn_argsort_rows(
    const VknBuffer* a, size_t a_off,
    int64_t rows, int64_t cols,
    int descending,
    VknBuffer* out_i32, size_t out_off,
    int dtype
) {
    if (!a || !out_i32 || rows <= 0 || cols <= 0) return -1;
    std::lock_guard<std::mutex> lock(g_mutex);
    if (ensure_init_locked() != 0) return -1;

    ggml_type gtype = to_ggml_type(dtype);
    ggml_context* ctx = make_graph_ctx();

    ggml_tensor* ta = ggml_new_tensor_2d(ctx, gtype, cols, rows);
    bind_tensor(ta, a, a_off);

    ggml_tensor* sa = ta;
    if (gtype != GGML_TYPE_F32) {
        ggml_tensor* sa_f32 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);
        sa = ggml_cpy(ctx, ta, sa_f32);
    }

    ggml_sort_order order = descending ? GGML_SORT_ORDER_DESC : GGML_SORT_ORDER_ASC;
    ggml_tensor* res = ggml_argsort(ctx, sa, order);
    bind_tensor(res, out_i32, out_off);

    int rc = run_graph(ctx, res);
    ggml_free(ctx);
    return rc;
}

int vkn_top_k_rows(
    const VknBuffer* a, size_t a_off,
    int64_t rows, int64_t cols,
    int k,
    VknBuffer* out_i32, size_t out_off,
    int dtype
) {
    if (!a || !out_i32 || rows <= 0 || cols <= 0 || k <= 0 || k > cols) return -1;
    std::lock_guard<std::mutex> lock(g_mutex);
    if (ensure_init_locked() != 0) return -1;

    ggml_type gtype = to_ggml_type(dtype);
    ggml_context* ctx = make_graph_ctx();

    ggml_tensor* ta = ggml_new_tensor_2d(ctx, gtype, cols, rows);
    bind_tensor(ta, a, a_off);

    ggml_tensor* sa = ta;
    if (gtype != GGML_TYPE_F32) {
        ggml_tensor* sa_f32 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);
        sa = ggml_cpy(ctx, ta, sa_f32);
    }

    ggml_tensor* res = ggml_top_k(ctx, sa, k);
    bind_tensor(res, out_i32, out_off);

    int rc = run_graph(ctx, res);
    ggml_free(ctx);
    return rc;
}

int vkn_get_rows(
    const VknBuffer* a, size_t a_off,
    int64_t rows, int64_t cols,
    const VknBuffer* idx_i32, size_t idx_off,
    int64_t n_idx,
    VknBuffer* out, size_t out_off,
    int dtype
) {
    if (!a || !idx_i32 || !out || rows <= 0 || cols <= 0 || n_idx <= 0) return -1;
    std::lock_guard<std::mutex> lock(g_mutex);
    if (ensure_init_locked() != 0) return -1;

    ggml_type gtype = to_ggml_type(dtype);
    ggml_context* ctx = make_graph_ctx();

    ggml_tensor* ta = ggml_new_tensor_2d(ctx, gtype, cols, rows);
    ggml_tensor* tidx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_idx);
    bind_tensor(ta, a, a_off);
    bind_tensor(tidx, idx_i32, idx_off);

    ggml_tensor* res = ggml_get_rows(ctx, ta, tidx); // F32 for F32/F16, I32 for I32
    ggml_tensor* final_node = res;

    if (res->type == gtype) {
        bind_tensor(res, out, out_off);
    } else {
        ggml_tensor* dst = ggml_new_tensor_2d(ctx, gtype, cols, n_idx);
        bind_tensor(dst, out, out_off);
        final_node = ggml_cpy(ctx, res, dst);
    }

    int rc = run_graph(ctx, final_node);
    ggml_free(ctx);
    return rc;
}

int vkn_solve_tri(
    const VknBuffer* a, size_t a_off, const int64_t* a_shape,
    const VknBuffer* b, size_t b_off, const int64_t* b_shape,
    int ndim,
    int left, int lower, int unit_diag,
    VknBuffer* out, size_t out_off,
    int dtype
) {
    if (!a || !b || !out || ndim < 2 || ndim > 4) return -1;
    std::lock_guard<std::mutex> lock(g_mutex);
    if (ensure_init_locked() != 0) return -1;

    ggml_type gtype = to_ggml_type(dtype);
    int64_t a_ne[4], b_ne[4];
    np_shape_to_ggml_ne(a_shape, ndim, a_ne);
    np_shape_to_ggml_ne(b_shape, ndim, b_ne);

    ggml_context* ctx = make_graph_ctx();
    ggml_tensor* ta = ggml_new_tensor_4d(ctx, gtype, a_ne[0], a_ne[1], a_ne[2], a_ne[3]);
    ggml_tensor* tb = ggml_new_tensor_4d(ctx, gtype, b_ne[0], b_ne[1], b_ne[2], b_ne[3]);
    bind_tensor(ta, a, a_off);
    bind_tensor(tb, b, b_off);

    ggml_tensor* sa = ta;
    ggml_tensor* sb = tb;
    if (gtype != GGML_TYPE_F32) {
        ggml_tensor* sa_f32 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, a_ne[0], a_ne[1], a_ne[2], a_ne[3]);
        ggml_tensor* sb_f32 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, b_ne[0], b_ne[1], b_ne[2], b_ne[3]);
        sa = ggml_cpy(ctx, ta, sa_f32);
        sb = ggml_cpy(ctx, tb, sb_f32);
    }

    ggml_tensor* res = ggml_solve_tri(ctx, sa, sb, left != 0, lower != 0, unit_diag != 0);
    ggml_tensor* final_node = res;

    if (gtype == GGML_TYPE_F32) {
        bind_tensor(res, out, out_off);
    } else {
        ggml_tensor* dst = ggml_new_tensor_4d(ctx, gtype, b_ne[0], b_ne[1], b_ne[2], b_ne[3]);
        bind_tensor(dst, out, out_off);
        final_node = ggml_cpy(ctx, res, dst);
    }

    int rc = run_graph(ctx, final_node);
    ggml_free(ctx);
    return rc;
}

int vkn_where(
    const VknBuffer* cond, size_t cond_off, const int64_t* cond_shape,
    const VknBuffer* x,    size_t x_off,    const int64_t* x_shape,
    const VknBuffer* y,    size_t y_off,    const int64_t* y_shape,
    VknBuffer* out,        size_t out_off,  const int64_t* out_shape,
    int ndim,
    int dtype
) {
    if (!cond || !x || !y || !out || ndim < 1 || ndim > 4) return -1;
    std::lock_guard<std::mutex> lock(g_mutex);
    if (ensure_init_locked() != 0) return -1;

    ggml_type gtype = to_ggml_type(dtype);
    int64_t c_ne[4], x_ne[4], y_ne[4], o_ne[4];
    np_shape_to_ggml_ne(cond_shape, ndim, c_ne);
    np_shape_to_ggml_ne(x_shape, ndim, x_ne);
    np_shape_to_ggml_ne(y_shape, ndim, y_ne);
    np_shape_to_ggml_ne(out_shape, ndim, o_ne);

    ggml_context* ctx = make_graph_ctx();
    ggml_tensor* tc = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c_ne[0], c_ne[1], c_ne[2], c_ne[3]);
    ggml_tensor* tx = ggml_new_tensor_4d(ctx, gtype, x_ne[0], x_ne[1], x_ne[2], x_ne[3]);
    ggml_tensor* ty = ggml_new_tensor_4d(ctx, gtype, y_ne[0], y_ne[1], y_ne[2], y_ne[3]);
    bind_tensor(tc, cond, cond_off);
    bind_tensor(tx, x, x_off);
    bind_tensor(ty, y, y_off);

    ggml_tensor* sx = tx;
    ggml_tensor* sy = ty;
    if (gtype != GGML_TYPE_F32) {
        ggml_tensor* sx_f32 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, x_ne[0], x_ne[1], x_ne[2], x_ne[3]);
        ggml_tensor* sy_f32 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, y_ne[0], y_ne[1], y_ne[2], y_ne[3]);
        sx = ggml_cpy(ctx, tx, sx_f32);
        sy = ggml_cpy(ctx, ty, sy_f32);
    }

    auto maybe_repeat = [&](ggml_tensor* t, const int64_t ne[4]) -> ggml_tensor* {
        if (ne[0] != o_ne[0] || ne[1] != o_ne[1] || ne[2] != o_ne[2] || ne[3] != o_ne[3]) {
            ggml_tensor* tmpl = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, o_ne[0], o_ne[1], o_ne[2], o_ne[3]);
            return ggml_repeat(ctx, t, tmpl);
        }
        return t;
    };

    // Normalize cond to {0.0, 1.0} mask
    ggml_tensor* mask = ggml_step(ctx, ggml_abs(ctx, tc));
    mask = maybe_repeat(mask, c_ne);
    ggml_tensor* inv_mask = ggml_scale_bias(ctx, mask, -1.0f, 1.0f);

    ggml_tensor* px = ggml_mul(ctx, mask, sx);
    ggml_tensor* py = ggml_mul(ctx, inv_mask, sy);
    ggml_tensor* res = ggml_add(ctx, px, py);
    ggml_tensor* final_node = res;

    if (gtype == GGML_TYPE_F32) {
        bind_tensor(res, out, out_off);
    } else {
        ggml_tensor* dst = ggml_new_tensor_4d(ctx, gtype, o_ne[0], o_ne[1], o_ne[2], o_ne[3]);
        bind_tensor(dst, out, out_off);
        final_node = ggml_cpy(ctx, res, dst);
    }

    int rc = run_graph(ctx, final_node);
    ggml_free(ctx);
    return rc;
}

int vkn_set_rows(
    VknBuffer* dst, size_t dst_off,
    int64_t rows, int64_t cols,
    const VknBuffer* src, size_t src_off,
    const VknBuffer* idx_i32, size_t idx_off,
    int64_t n_idx,
    int dtype
) {
    if (!dst || !src || !idx_i32 || rows <= 0 || cols <= 0 || n_idx <= 0) return -1;
    std::lock_guard<std::mutex> lock(g_mutex);
    if (ensure_init_locked() != 0) return -1;

    ggml_type gtype = to_ggml_type(dtype);
    ggml_context* ctx = make_graph_ctx();

    ggml_tensor* td = ggml_new_tensor_2d(ctx, gtype, cols, rows);
    ggml_tensor* ts = ggml_new_tensor_2d(ctx, gtype, cols, n_idx);
    ggml_tensor* ti = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_idx);
    bind_tensor(td, dst, dst_off);
    bind_tensor(ts, src, src_off);
    bind_tensor(ti, idx_i32, idx_off);

    ggml_tensor* final_node = nullptr;
    if (gtype == GGML_TYPE_F32) {
        final_node = ggml_set_rows(ctx, td, ts, ti);
    } else {
        ggml_tensor* td_f32 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);
        ggml_tensor* ts_f32 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, n_idx);
        ggml_tensor* cd = ggml_cpy(ctx, td, td_f32);
        ggml_tensor* cs = ggml_cpy(ctx, ts, ts_f32);
        ggml_tensor* scattered = ggml_set_rows(ctx, cd, cs, ti);
        final_node = ggml_cpy(ctx, scattered, td);
    }

    int rc = run_graph(ctx, final_node);
    ggml_free(ctx);
    return rc;
}

int vkn_scatter_nonzero(
    const VknBuffer* mask_f32, size_t mask_off,
    const VknBuffer* cumsum_f32, size_t cumsum_off,
    int64_t n, int64_t k,
    VknBuffer* out_idx_i32, size_t out_off
) {
    if (!mask_f32 || !cumsum_f32 || !out_idx_i32 || n <= 0 || k <= 0) return -1;
    std::lock_guard<std::mutex> lock(g_mutex);
    if (ensure_init_locked() != 0) return -1;

    ggml_context* ctx = make_graph_ctx();

    ggml_tensor* tm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor* tc = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    bind_tensor(tm, mask_f32, mask_off);
    bind_tensor(tc, cumsum_f32, cumsum_off);

    // mask in {0, 1}; dest_pos = mask * (cumsum - 1) + (1 - mask) * k
    ggml_tensor* mask01 = ggml_step(ctx, ggml_abs(ctx, tm));
    ggml_tensor* inv_mask = ggml_scale_bias(ctx, mask01, -1.0f, 1.0f);
    ggml_tensor* pos_true = ggml_mul(ctx, mask01, ggml_scale_bias(ctx, tc, 1.0f, -1.0f));
    ggml_tensor* pos_false = ggml_scale(ctx, inv_mask, static_cast<float>(k));
    ggml_tensor* dest_f32 = ggml_add(ctx, pos_true, pos_false);

    ggml_tensor* dest_i32 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
    ggml_tensor* idx_i32 = ggml_cpy(ctx, dest_f32, dest_i32);

    // Source row IDs 0..n-1 as [1, n]
    ggml_tensor* ar = ggml_arange(ctx, 0.0f, static_cast<float>(n), 1.0f);
    ggml_tensor* row_ids = ggml_reshape_2d(ctx, ar, 1, n);

    // Scratch destination [1, k + 1] (slot k is dummy sink for False elements)
    ggml_tensor* sink = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, k + 1);
    ggml_tensor* filled_sink = ggml_fill_inplace(ctx, sink, 0.0f);
    ggml_tensor* scattered = ggml_set_rows(ctx, filled_sink, row_ids, idx_i32);

    // View first k elements and cast F32 -> I32 directly into out_idx_i32
    ggml_tensor* first_k = ggml_view_1d(ctx, scattered, k, 0);
    ggml_tensor* out_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, k);
    bind_tensor(out_t, out_idx_i32, out_off);
    ggml_tensor* final_node = ggml_cpy(ctx, first_k, out_t);

    int rc = run_graph(ctx, final_node);
    ggml_free(ctx);
    return rc;
}

} // extern "C"

/* Internal C++ accessors for zero-copy ggml-vulkan direct shaders */
ggml_backend_t vkn_internal_get_backend() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (ensure_init_locked() != 0) return nullptr;
    return g_backend;
}

ggml_backend_buffer_t vkn_internal_get_buffer(const VknBuffer* buf) {
    return buf ? buf->buf : nullptr;
}
