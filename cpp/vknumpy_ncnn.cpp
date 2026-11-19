#include "vknumpy_ggml.h"

#ifdef VKNUMPY_USE_NCNN

#include "gpu.h"
#include "layer.h"
#include "mat.h"
#include "option.h"
#include "paramdict.h"

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

} // extern "C"

#else

extern "C" {
int vkn_ncnn_available(void) { return 0; }
int vkn_ncnn_unary(int, const VknBuffer*, size_t, int64_t, VknBuffer*, size_t, int) { return -1; }
int vkn_ncnn_binary(int, const VknBuffer*, size_t, const VknBuffer*, size_t, int64_t, VknBuffer*, size_t, int) { return -1; }
int vkn_ncnn_reduce_rows(int, const VknBuffer*, size_t, int64_t, int64_t, VknBuffer*, size_t, int) { return -1; }
}

#endif
