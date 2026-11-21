#include "vknumpy_ggml.h"

#ifdef VKNUMPY_USE_NCNN

#include "gpu.h"
#include "layer.h"
#include "mat.h"
#include "option.h"
#include "paramdict.h"
#include "pipeline.h"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace {

std::mutex g_ncnn_mutex;
bool g_ncnn_ready = false;
ncnn::VulkanDevice* g_vkdev = nullptr;
ncnn::VkAllocator* g_blob_alloc = nullptr;
ncnn::VkAllocator* g_staging_alloc = nullptr;

std::unordered_map<int, ncnn::Layer*> g_unary_layers;
std::unordered_map<int, ncnn::Layer*> g_binary_layers;
std::unordered_map<int, ncnn::Layer*> g_reduce_layers;

ncnn::Pipeline* g_pipe_bincount_i32 = nullptr;
ncnn::Pipeline* g_pipe_bincount_wf32 = nullptr;
ncnn::Pipeline* g_pipe_hist2d_f32 = nullptr;

ncnn::Pipeline* build_compute_pipeline_locked(const char* glsl_src, const ncnn::Option& opt) {
    std::vector<uint32_t> spirv;
    if (ncnn::compile_spirv_module(glsl_src, opt, spirv) != 0 || spirv.empty()) {
        return nullptr;
    }
    ncnn::Pipeline* pipe = new ncnn::Pipeline(g_vkdev);
    pipe->set_local_size_xyz(256, 1, 1);
    std::vector<ncnn::vk_specialization_type> specs;
    if (pipe->create(spirv.data(), spirv.size() * sizeof(uint32_t), specs) != 0) {
        delete pipe;
        return nullptr;
    }
    return pipe;
}

ncnn::Pipeline* get_bincount_i32_pipeline_locked(const ncnn::Option& opt) {
    if (g_pipe_bincount_i32) return g_pipe_bincount_i32;
    static const char* kGlsl = R"(
#version 450
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(binding = 0) readonly buffer XBuf { int xs[]; };
layout(binding = 1) buffer OutBuf { int counts[]; };
layout(push_constant) uniform parameter {
    int n;
    int out_len;
} p;
void main() {
    uint stride = gl_NumWorkGroups.x * 256u;
    for (uint i = gl_GlobalInvocationID.x; i < uint(p.n); i += stride) {
        int b = xs[i];
        if (b >= 0 && b < p.out_len) {
            atomicAdd(counts[b], 1);
        }
    }
}
)";
    g_pipe_bincount_i32 = build_compute_pipeline_locked(kGlsl, opt);
    return g_pipe_bincount_i32;
}

ncnn::Pipeline* get_bincount_wf32_pipeline_locked(const ncnn::Option& opt) {
    if (g_pipe_bincount_wf32) return g_pipe_bincount_wf32;
    static const char* kGlsl = R"(
#version 450
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(binding = 0) readonly buffer XBuf { int xs[]; };
layout(binding = 1) readonly buffer WBuf { float ws[]; };
layout(binding = 2) buffer OutBuf { uint sums[]; };
layout(push_constant) uniform parameter {
    int n;
    int out_len;
} p;
void main() {
    uint stride = gl_NumWorkGroups.x * 256u;
    for (uint i = gl_GlobalInvocationID.x; i < uint(p.n); i += stride) {
        int b = xs[i];
        if (b >= 0 && b < p.out_len) {
            float w = ws[i];
            uint old_bits = sums[b];
            for (;;) {
                uint new_bits = floatBitsToUint(uintBitsToFloat(old_bits) + w);
                uint prev = atomicCompSwap(sums[b], old_bits, new_bits);
                if (prev == old_bits) break;
                old_bits = prev;
            }
        }
    }
}
)";
    g_pipe_bincount_wf32 = build_compute_pipeline_locked(kGlsl, opt);
    return g_pipe_bincount_wf32;
}

ncnn::Pipeline* get_hist2d_f32_pipeline_locked(const ncnn::Option& opt) {
    if (g_pipe_hist2d_f32) return g_pipe_hist2d_f32;
    static const char* kGlsl = R"(
#version 450
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(binding = 0) readonly buffer XBuf { float xs[]; };
layout(binding = 1) readonly buffer YBuf { float ys[]; };
layout(binding = 2) buffer OutBuf { int grid[]; };
layout(push_constant) uniform parameter {
    int n;
    int w;
    int h;
    int flip_y;
    float xmin;
    float inv_dx;
    float ymin;
    float ymax;
    float inv_dy;
} p;
void main() {
    uint stride = gl_NumWorkGroups.x * 256u;
    for (uint i = gl_GlobalInvocationID.x; i < uint(p.n); i += stride) {
        float xv = xs[i];
        float yv = ys[i];
        float u = (xv - p.xmin) * p.inv_dx;
        float v = (p.flip_y != 0) ? ((p.ymax - yv) * p.inv_dy) : ((yv - p.ymin) * p.inv_dy);
        if (u >= 0.0 && u < 1.0 && v >= 0.0 && v < 1.0) {
            int px = clamp(int(u * float(p.w)), 0, p.w - 1);
            int py = clamp(int(v * float(p.h)), 0, p.h - 1);
            atomicAdd(grid[py * p.w + px], 1);
        }
    }
}
)";
    g_pipe_hist2d_f32 = build_compute_pipeline_locked(kGlsl, opt);
    return g_pipe_hist2d_f32;
}

ncnn::Option make_opt() {
    ncnn::Option opt;
    opt.use_vulkan_compute = true;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_packing_layout = false;
    opt.blob_vkallocator = g_blob_alloc;
    opt.workspace_vkallocator = g_blob_alloc;
    opt.staging_vkallocator = g_staging_alloc;
    return opt;
}

bool ensure_ncnn_locked() {
    if (g_ncnn_ready) return g_vkdev != nullptr;
    g_ncnn_ready = true;
    if (ncnn::get_gpu_count() <= 0) {
        return false;
    }
    g_vkdev = ncnn::get_gpu_device(0);
    if (!g_vkdev) return false;
    g_blob_alloc = g_vkdev->acquire_blob_allocator();
    g_staging_alloc = g_vkdev->acquire_staging_allocator();
    return true;
}

ncnn::Layer* get_unary_layer_locked(int op_type, const ncnn::Option& opt) {
    auto it = g_unary_layers.find(op_type);
    if (it != g_unary_layers.end()) return it->second;

    ncnn::Layer* layer = ncnn::create_layer("UnaryOp");
    if (!layer) return nullptr;
    layer->vkdev = g_vkdev;
    ncnn::ParamDict pd;
    pd.set(0, op_type);
    layer->load_param(pd);
    layer->create_pipeline(opt);
    g_unary_layers[op_type] = layer;
    return layer;
}

ncnn::Layer* get_binary_layer_locked(int op_type, const ncnn::Option& opt) {
    auto it = g_binary_layers.find(op_type);
    if (it != g_binary_layers.end()) return it->second;

    ncnn::Layer* layer = ncnn::create_layer("BinaryOp");
    if (!layer) return nullptr;
    layer->vkdev = g_vkdev;
    ncnn::ParamDict pd;
    pd.set(0, op_type);
    pd.set(1, 0); // with_scalar = 0
    layer->load_param(pd);
    layer->create_pipeline(opt);
    g_binary_layers[op_type] = layer;
    return layer;
}

ncnn::Layer* get_reduce_layer_locked(int reduce_op, const ncnn::Option& opt) {
    auto it = g_reduce_layers.find(reduce_op);
    if (it != g_reduce_layers.end()) return it->second;

    ncnn::Layer* layer = ncnn::create_layer("Reduction");
    if (!layer) return nullptr;
    layer->vkdev = g_vkdev;
    ncnn::ParamDict pd;
    pd.set(0, reduce_op);
    pd.set(1, 0); // reduce_all = 0
    pd.set(2, 1.0f); // coeff = 1.0
    ncnn::Mat axes(1);
    ((int*)axes)[0] = -1; // reduce along last axis (W = cols in 2D Mat(w=cols, h=rows))
    pd.set(3, axes);
    pd.set(4, 0); // keepdims = 0
    pd.set(5, 1); // fixbug0 = 1 (standard 0-based axes without legacy batch offset)
    layer->load_param(pd);
    layer->create_pipeline(opt);
    g_reduce_layers[reduce_op] = layer;
    return layer;
}

} // namespace

extern "C" {

int vkn_ncnn_available(void) {
    std::lock_guard<std::mutex> lock(g_ncnn_mutex);
    return ensure_ncnn_locked() ? 1 : 0;
}

int vkn_ncnn_unary(
    int ncnn_op_type,
    const VknBuffer* a, size_t a_off,
    int64_t n,
    VknBuffer* out, size_t out_off,
    int dtype
) {
    if (!a || !out || n <= 0 || dtype != VKN_DTYPE_F32) return -1;
    std::lock_guard<std::mutex> lock(g_ncnn_mutex);
    if (!ensure_ncnn_locked()) return -1;

    ncnn::Option opt = make_opt();
    ncnn::Layer* layer = get_unary_layer_locked(ncnn_op_type, opt);
    if (!layer) return -1;

    ncnn::Mat host_in(static_cast<int>(n));
    if (vkn_buf_download(a, a_off, host_in.data, static_cast<size_t>(n) * sizeof(float)) != 0) {
        return -1;
    }

    ncnn::VkCompute cmd(g_vkdev);
    ncnn::VkMat gpu_mat;
    cmd.record_clone(host_in, gpu_mat, opt);
    if (layer->forward_inplace(gpu_mat, cmd, opt) != 0) return -1;
    ncnn::Mat host_out;
    cmd.record_clone(gpu_mat, host_out, opt);
    if (cmd.submit_and_wait() != 0) return -1;

    return vkn_buf_upload(out, out_off, host_out.data, static_cast<size_t>(n) * sizeof(float));
}

int vkn_ncnn_binary(
    int ncnn_op_type,
    const VknBuffer* a, size_t a_off,
    const VknBuffer* b, size_t b_off,
    int64_t n,
    VknBuffer* out, size_t out_off,
    int dtype
) {
    if (!a || !b || !out || n <= 0 || dtype != VKN_DTYPE_F32) return -1;
    std::lock_guard<std::mutex> lock(g_ncnn_mutex);
    if (!ensure_ncnn_locked()) return -1;

    ncnn::Option opt = make_opt();
    ncnn::Layer* layer = get_binary_layer_locked(ncnn_op_type, opt);
    if (!layer) return -1;

    ncnn::Mat host_a(static_cast<int>(n));
    ncnn::Mat host_b(static_cast<int>(n));
    if (vkn_buf_download(a, a_off, host_a.data, static_cast<size_t>(n) * sizeof(float)) != 0) return -1;
    if (vkn_buf_download(b, b_off, host_b.data, static_cast<size_t>(n) * sizeof(float)) != 0) return -1;

    ncnn::VkCompute cmd(g_vkdev);
    ncnn::VkMat gpu_a, gpu_b;
    cmd.record_clone(host_a, gpu_a, opt);
    cmd.record_clone(host_b, gpu_b, opt);

    std::vector<ncnn::VkMat> bottoms = {gpu_a, gpu_b};
    std::vector<ncnn::VkMat> tops(1);
    if (layer->forward(bottoms, tops, cmd, opt) != 0) return -1;

    ncnn::Mat host_out;
    cmd.record_clone(tops[0], host_out, opt);
    if (cmd.submit_and_wait() != 0) return -1;

    return vkn_buf_upload(out, out_off, host_out.data, static_cast<size_t>(n) * sizeof(float));
}

int vkn_ncnn_reduce_rows(
    int ncnn_reduce_op,
    const VknBuffer* a, size_t a_off,
    int64_t rows, int64_t cols,
    VknBuffer* out, size_t out_off,
    int dtype
) {
    if (!a || !out || rows <= 0 || cols <= 0 || dtype != VKN_DTYPE_F32) return -1;
    std::lock_guard<std::mutex> lock(g_ncnn_mutex);
    if (!ensure_ncnn_locked()) return -1;

    ncnn::Option opt = make_opt();
    ncnn::Layer* layer = get_reduce_layer_locked(ncnn_reduce_op, opt);
    if (!layer) return -1;

    ncnn::Mat host_in(static_cast<int>(cols), static_cast<int>(rows));
    if (vkn_buf_download(a, a_off, host_in.data, static_cast<size_t>(rows * cols) * sizeof(float)) != 0) {
        return -1;
    }

    ncnn::VkCompute cmd(g_vkdev);
    ncnn::VkMat gpu_in, gpu_out;
    cmd.record_clone(host_in, gpu_in, opt);
    if (layer->forward(gpu_in, gpu_out, cmd, opt) != 0) return -1;

    ncnn::Mat host_out;
    cmd.record_clone(gpu_out, host_out, opt);
    if (cmd.submit_and_wait() != 0) return -1;

    return vkn_buf_upload(out, out_off, host_out.data, static_cast<size_t>(rows) * sizeof(float));
}

int vkn_ncnn_bincount_i32(
    const VknBuffer* x_i32, size_t x_off,
    int64_t n,
    VknBuffer* out_i32, size_t out_off,
    int64_t out_len
) {
    if (!x_i32 || !out_i32 || n < 0 || out_len <= 0) return -1;
    if (n == 0) {
        std::vector<int32_t> zeros(static_cast<size_t>(out_len), 0);
        return vkn_buf_upload(out_i32, out_off, zeros.data(), zeros.size() * sizeof(int32_t));
    }
    std::lock_guard<std::mutex> lock(g_ncnn_mutex);
    if (!ensure_ncnn_locked()) return -1;

    ncnn::Option opt = make_opt();
    ncnn::Pipeline* pipe = get_bincount_i32_pipeline_locked(opt);
    if (!pipe) return -1;

    ncnn::Mat host_x(static_cast<int>(n), size_t(4));
    if (vkn_buf_download(x_i32, x_off, host_x.data, static_cast<size_t>(n) * sizeof(int32_t)) != 0) {
        return -1;
    }
    ncnn::Mat host_out(static_cast<int>(out_len), size_t(4));
    std::memset(host_out.data, 0, static_cast<size_t>(out_len) * sizeof(int32_t));

    ncnn::VkCompute cmd(g_vkdev);
    ncnn::VkMat gpu_x, gpu_out;
    cmd.record_clone(host_x, gpu_x, opt);
    cmd.record_clone(host_out, gpu_out, opt);

    std::vector<ncnn::VkMat> bindings = { gpu_x, gpu_out };
    std::vector<ncnn::vk_constant_type> constants(2);
    constants[0].i = static_cast<int>(n);
    constants[1].i = static_cast<int>(out_len);

    int dispatch_threads = static_cast<int>(std::min<int64_t>(n, 256LL * 32768LL));
    ncnn::Mat dispatcher(dispatch_threads, 1, 1, (void*)0);
    cmd.record_pipeline(pipe, bindings, std::vector<ncnn::VkImageMat>(), constants, dispatcher);

    ncnn::Mat res_out;
    cmd.record_clone(gpu_out, res_out, opt);
    if (cmd.submit_and_wait() != 0) return -1;

    return vkn_buf_upload(out_i32, out_off, res_out.data, static_cast<size_t>(out_len) * sizeof(int32_t));
}

int vkn_ncnn_bincount_weighted_f32(
    const VknBuffer* x_i32, size_t x_off,
    const VknBuffer* w_f32, size_t w_off,
    int64_t n,
    VknBuffer* out_f32, size_t out_off,
    int64_t out_len
) {
    if (!x_i32 || !w_f32 || !out_f32 || n < 0 || out_len <= 0) return -1;
    if (n == 0) {
        std::vector<float> zeros(static_cast<size_t>(out_len), 0.0f);
        return vkn_buf_upload(out_f32, out_off, zeros.data(), zeros.size() * sizeof(float));
    }
    std::lock_guard<std::mutex> lock(g_ncnn_mutex);
    if (!ensure_ncnn_locked()) return -1;

    ncnn::Option opt = make_opt();
    ncnn::Pipeline* pipe = get_bincount_wf32_pipeline_locked(opt);
    if (!pipe) return -1;

    ncnn::Mat host_x(static_cast<int>(n), size_t(4));
    ncnn::Mat host_w(static_cast<int>(n), size_t(4));
    if (vkn_buf_download(x_i32, x_off, host_x.data, static_cast<size_t>(n) * sizeof(int32_t)) != 0 ||
        vkn_buf_download(w_f32, w_off, host_w.data, static_cast<size_t>(n) * sizeof(float)) != 0) {
        return -1;
    }
    ncnn::Mat host_out(static_cast<int>(out_len), size_t(4));
    std::memset(host_out.data, 0, static_cast<size_t>(out_len) * sizeof(float));

    ncnn::VkCompute cmd(g_vkdev);
    ncnn::VkMat gpu_x, gpu_w, gpu_out;
    cmd.record_clone(host_x, gpu_x, opt);
    cmd.record_clone(host_w, gpu_w, opt);
    cmd.record_clone(host_out, gpu_out, opt);

    std::vector<ncnn::VkMat> bindings = { gpu_x, gpu_w, gpu_out };
    std::vector<ncnn::vk_constant_type> constants(2);
    constants[0].i = static_cast<int>(n);
    constants[1].i = static_cast<int>(out_len);

    int dispatch_threads = static_cast<int>(std::min<int64_t>(n, 256LL * 32768LL));
    ncnn::Mat dispatcher(dispatch_threads, 1, 1, (void*)0);
    cmd.record_pipeline(pipe, bindings, std::vector<ncnn::VkImageMat>(), constants, dispatcher);

    ncnn::Mat res_out;
    cmd.record_clone(gpu_out, res_out, opt);
    if (cmd.submit_and_wait() != 0) return -1;

    return vkn_buf_upload(out_f32, out_off, res_out.data, static_cast<size_t>(out_len) * sizeof(float));
}

int vkn_ncnn_histogram2d_f32(
    const VknBuffer* x_f32, size_t x_off,
    const VknBuffer* y_f32, size_t y_off,
    int64_t n,
    float xmin, float xmax,
    float ymin, float ymax,
    int width, int height,
    int flip_y,
    VknBuffer* out_i32, size_t out_off
) {
    if (!x_f32 || !y_f32 || !out_i32 || n < 0 || width <= 0 || height <= 0) return -1;
    if (!(xmax > xmin) || !(ymax > ymin)) return -1;
    int64_t grid_len = static_cast<int64_t>(width) * static_cast<int64_t>(height);
    if (n == 0) {
        std::vector<int32_t> zeros(static_cast<size_t>(grid_len), 0);
        return vkn_buf_upload(out_i32, out_off, zeros.data(), zeros.size() * sizeof(int32_t));
    }
    std::lock_guard<std::mutex> lock(g_ncnn_mutex);
    if (!ensure_ncnn_locked()) return -1;

    ncnn::Option opt = make_opt();
    ncnn::Pipeline* pipe = get_hist2d_f32_pipeline_locked(opt);
    if (!pipe) return -1;

    ncnn::Mat host_x(static_cast<int>(n), size_t(4));
    ncnn::Mat host_y(static_cast<int>(n), size_t(4));
    if (vkn_buf_download(x_f32, x_off, host_x.data, static_cast<size_t>(n) * sizeof(float)) != 0 ||
        vkn_buf_download(y_f32, y_off, host_y.data, static_cast<size_t>(n) * sizeof(float)) != 0) {
        return -1;
    }
    ncnn::Mat host_out(static_cast<int>(grid_len), size_t(4));
    std::memset(host_out.data, 0, static_cast<size_t>(grid_len) * sizeof(int32_t));

    ncnn::VkCompute cmd(g_vkdev);
    ncnn::VkMat gpu_x, gpu_y, gpu_out;
    cmd.record_clone(host_x, gpu_x, opt);
    cmd.record_clone(host_y, gpu_y, opt);
    cmd.record_clone(host_out, gpu_out, opt);

    std::vector<ncnn::VkMat> bindings = { gpu_x, gpu_y, gpu_out };
    std::vector<ncnn::vk_constant_type> constants(9);
    constants[0].i = static_cast<int>(n);
    constants[1].i = width;
    constants[2].i = height;
    constants[3].i = flip_y;
    constants[4].f = xmin;
    constants[5].f = 1.0f / (xmax - xmin);
    constants[6].f = ymin;
    constants[7].f = ymax;
    constants[8].f = 1.0f / (ymax - ymin);

    int dispatch_threads = static_cast<int>(std::min<int64_t>(n, 256LL * 32768LL));
    ncnn::Mat dispatcher(dispatch_threads, 1, 1, (void*)0);
    cmd.record_pipeline(pipe, bindings, std::vector<ncnn::VkImageMat>(), constants, dispatcher);

    ncnn::Mat res_out;
    cmd.record_clone(gpu_out, res_out, opt);
    if (cmd.submit_and_wait() != 0) return -1;

    return vkn_buf_upload(out_i32, out_off, res_out.data, static_cast<size_t>(grid_len) * sizeof(int32_t));
}

} // extern "C"

#else

extern "C" {
int vkn_ncnn_available(void) { return 0; }
int vkn_ncnn_unary(int, const VknBuffer*, size_t, int64_t, VknBuffer*, size_t, int) { return -1; }
int vkn_ncnn_binary(int, const VknBuffer*, size_t, const VknBuffer*, size_t, int64_t, VknBuffer*, size_t, int) { return -1; }
int vkn_ncnn_reduce_rows(int, const VknBuffer*, size_t, int64_t, int64_t, VknBuffer*, size_t, int) { return -1; }
int vkn_ncnn_bincount_i32(const VknBuffer*, size_t, int64_t, VknBuffer*, size_t, int64_t) { return -1; }
int vkn_ncnn_bincount_weighted_f32(const VknBuffer*, size_t, const VknBuffer*, size_t, int64_t, VknBuffer*, size_t, int64_t) { return -1; }
int vkn_ncnn_histogram2d_f32(const VknBuffer*, size_t, const VknBuffer*, size_t, int64_t, float, float, float, float, int, int, int, VknBuffer*, size_t) { return -1; }
}

#endif
