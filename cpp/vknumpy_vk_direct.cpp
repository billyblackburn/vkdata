#include "vknumpy_ggml.h"

#include "ggml-backend.h"
#include "ggml-vulkan.h"

#include <vulkan/vulkan.h>

#include "vkn_spv_bincount_i32.h"
#include "vkn_spv_bincount_wf32.h"
#include "vkn_spv_histogram2d_f32.h"
#include "vkn_spv_histogram1d_f32.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>

ggml_backend_t vkn_internal_get_backend();
ggml_backend_buffer_t vkn_internal_get_buffer(const VknBuffer* buf);

extern "C" {
void*    ggml_backend_vk_get_raw_device(ggml_backend_t backend);
uint32_t ggml_backend_vk_get_compute_queue_family(ggml_backend_t backend);
void*    ggml_backend_vk_get_raw_buffer(ggml_backend_buffer_t buffer);
int      ggml_backend_vk_submit_compute_cmd(ggml_backend_t backend, void* cmd_buf, void* fence);
}

namespace {

struct DirectPipeline {
    VkShaderModule        shader_module = VK_NULL_HANDLE;
    VkDescriptorSetLayout desc_layout   = VK_NULL_HANDLE;
    VkPipelineLayout      pipe_layout   = VK_NULL_HANDLE;
    VkPipeline            pipeline      = VK_NULL_HANDLE;
    VkDescriptorSet       desc_set      = VK_NULL_HANDLE;
};

struct DirectState {
    bool             initialized  = false;
    bool             ok           = false;
    VkDevice         dev          = VK_NULL_HANDLE;
    uint32_t         queue_family = UINT32_MAX;
    VkCommandPool    cmd_pool     = VK_NULL_HANDLE;
    VkCommandBuffer  cmd_buf      = VK_NULL_HANDLE;
    VkFence          fence        = VK_NULL_HANDLE;
    VkDescriptorPool desc_pool    = VK_NULL_HANDLE;

    DirectPipeline p_bincount_i32;
    DirectPipeline p_bincount_wf32;
    DirectPipeline p_hist2d_f32;
    DirectPipeline p_hist1d_f32;
};

std::mutex g_direct_mutex;
DirectState g_state;

bool create_pipeline(
    DirectState& st,
    const uint32_t* spv_words,
    size_t spv_bytes,
    uint32_t num_buffers,
    uint32_t push_constant_bytes,
    DirectPipeline& out
) {
    VkShaderModuleCreateInfo sm_ci{};
    sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    sm_ci.codeSize = spv_bytes;
    sm_ci.pCode = spv_words;
    if (vkCreateShaderModule(st.dev, &sm_ci, nullptr, &out.shader_module) != VK_SUCCESS) {
        return false;
    }

    std::vector<VkDescriptorSetLayoutBinding> bindings(num_buffers);
    for (uint32_t i = 0; i < num_buffers; ++i) {
        bindings[i] = {};
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo dsl_ci{};
    dsl_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl_ci.bindingCount = num_buffers;
    dsl_ci.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(st.dev, &dsl_ci, nullptr, &out.desc_layout) != VK_SUCCESS) {
        return false;
    }

    VkPushConstantRange pc_range{};
    pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc_range.offset = 0;
    pc_range.size = push_constant_bytes;

    VkPipelineLayoutCreateInfo pl_ci{};
    pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl_ci.setLayoutCount = 1;
    pl_ci.pSetLayouts = &out.desc_layout;
    pl_ci.pushConstantRangeCount = (push_constant_bytes > 0) ? 1 : 0;
    pl_ci.pPushConstantRanges = (push_constant_bytes > 0) ? &pc_range : nullptr;
    if (vkCreatePipelineLayout(st.dev, &pl_ci, nullptr, &out.pipe_layout) != VK_SUCCESS) {
        return false;
    }

    VkComputePipelineCreateInfo cp_ci{};
    cp_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cp_ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cp_ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cp_ci.stage.module = out.shader_module;
    cp_ci.stage.pName = "main";
    cp_ci.layout = out.pipe_layout;
    if (vkCreateComputePipelines(st.dev, VK_NULL_HANDLE, 1, &cp_ci, nullptr, &out.pipeline) != VK_SUCCESS) {
        return false;
    }

    VkDescriptorSetAllocateInfo dsa_ci{};
    dsa_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsa_ci.descriptorPool = st.desc_pool;
    dsa_ci.descriptorSetCount = 1;
    dsa_ci.pSetLayouts = &out.desc_layout;
    if (vkAllocateDescriptorSets(st.dev, &dsa_ci, &out.desc_set) != VK_SUCCESS) {
        return false;
    }
    return true;
}

bool ensure_direct_locked(ggml_backend_t backend) {
    if (!backend || !ggml_backend_is_vk(backend)) {
        return false;
    }
    if (g_state.initialized) {
        return g_state.ok;
    }
    g_state.initialized = true;

    g_state.dev = static_cast<VkDevice>(ggml_backend_vk_get_raw_device(backend));
    g_state.queue_family = ggml_backend_vk_get_compute_queue_family(backend);
    if (g_state.dev == VK_NULL_HANDLE || g_state.queue_family == UINT32_MAX) {
        g_state.ok = false;
        return false;
    }

    VkCommandPoolCreateInfo cp_ci{};
    cp_ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cp_ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cp_ci.queueFamilyIndex = g_state.queue_family;
    if (vkCreateCommandPool(g_state.dev, &cp_ci, nullptr, &g_state.cmd_pool) != VK_SUCCESS) {
        return false;
    }

    VkCommandBufferAllocateInfo cb_ai{};
    cb_ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cb_ai.commandPool = g_state.cmd_pool;
    cb_ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cb_ai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(g_state.dev, &cb_ai, &g_state.cmd_buf) != VK_SUCCESS) {
        return false;
    }

    VkFenceCreateInfo f_ci{};
    f_ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (vkCreateFence(g_state.dev, &f_ci, nullptr, &g_state.fence) != VK_SUCCESS) {
        return false;
    }

    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = 24;

    VkDescriptorPoolCreateInfo dp_ci{};
    dp_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dp_ci.maxSets = 6;
    dp_ci.poolSizeCount = 1;
    dp_ci.pPoolSizes = &pool_size;
    if (vkCreateDescriptorPool(g_state.dev, &dp_ci, nullptr, &g_state.desc_pool) != VK_SUCCESS) {
        return false;
    }

    bool ok1 = create_pipeline(
        g_state,
        vkn_spv_bincount_i32, sizeof(vkn_spv_bincount_i32),
        2, 4 * sizeof(uint32_t),
        g_state.p_bincount_i32
    );
    bool ok2 = create_pipeline(
        g_state,
        vkn_spv_bincount_wf32, sizeof(vkn_spv_bincount_wf32),
        3, 5 * sizeof(uint32_t),
        g_state.p_bincount_wf32
    );
    bool ok3 = create_pipeline(
        g_state,
        vkn_spv_histogram2d_f32, sizeof(vkn_spv_histogram2d_f32),
        3, 12 * sizeof(uint32_t),
        g_state.p_hist2d_f32
    );
    bool ok4 = create_pipeline(
        g_state,
        vkn_spv_histogram1d_f32, sizeof(vkn_spv_histogram1d_f32),
        4, 14 * sizeof(uint32_t),
        g_state.p_hist1d_f32
    );

    g_state.ok = ok1 && ok2 && ok3 && ok4;
    return g_state.ok;
}

VkBuffer get_vk_buffer(const VknBuffer* buf) {
    ggml_backend_buffer_t gbuf = vkn_internal_get_buffer(buf);
    if (!gbuf) return VK_NULL_HANDLE;
    return static_cast<VkBuffer>(ggml_backend_vk_get_raw_buffer(gbuf));
}

int dispatch_direct(
    ggml_backend_t backend,
    const DirectPipeline& pipe,
    const std::vector<VkBuffer>& buffers,
    VkBuffer out_buf,
    size_t out_off,
    size_t out_bytes,
    const void* pc_data,
    uint32_t pc_bytes,
    int64_t n
) {
    ggml_backend_synchronize(backend);

    std::vector<VkDescriptorBufferInfo> dbis(buffers.size());
    std::vector<VkWriteDescriptorSet> writes(buffers.size());
    for (uint32_t i = 0; i < buffers.size(); ++i) {
        dbis[i] = { buffers[i], 0, VK_WHOLE_SIZE };
        writes[i] = {};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = pipe.desc_set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &dbis[i];
    }
    vkUpdateDescriptorSets(g_state.dev, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    VkCommandBuffer cmd = g_state.cmd_buf;
    if (vkResetCommandBuffer(cmd, 0) != VK_SUCCESS) return -1;

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS) return -1;

    // Zero output buffer directly on GPU (zero host memset / upload)
    vkCmdFillBuffer(cmd, out_buf, out_off, out_bytes, 0);

    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1, &barrier,
        0, nullptr,
        0, nullptr
    );

    if (n > 0) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipe_layout, 0, 1, &pipe.desc_set, 0, nullptr);
        vkCmdPushConstants(cmd, pipe.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pc_bytes, pc_data);

        uint32_t groups_x = static_cast<uint32_t>(std::min<int64_t>((n + 255) / 256, 32768));
        vkCmdDispatch(cmd, groups_x, 1, 1);
    }

    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) return -1;

    if (ggml_backend_vk_submit_compute_cmd(backend, cmd, g_state.fence) != 0) {
        return -1;
    }
    VkResult wrc = vkWaitForFences(g_state.dev, 1, &g_state.fence, VK_TRUE, UINT64_MAX);
    VkResult rrc = vkResetFences(g_state.dev, 1, &g_state.fence);
    return (wrc == VK_SUCCESS && rrc == VK_SUCCESS) ? 0 : -1;
}

} // namespace

extern "C" {

int vkn_vk_direct_available(void) {
    std::lock_guard<std::mutex> lock(g_direct_mutex);
    ggml_backend_t backend = vkn_internal_get_backend();
    return ensure_direct_locked(backend) ? 1 : 0;
}

void* vkn_vk_get_device(void) {
    ggml_backend_t backend = vkn_internal_get_backend();
    return ggml_backend_vk_get_raw_device(backend);
}

uint32_t vkn_vk_get_compute_queue_family(void) {
    ggml_backend_t backend = vkn_internal_get_backend();
    return ggml_backend_vk_get_compute_queue_family(backend);
}

void* vkn_vk_get_raw_buffer(const VknBuffer* buf) {
    ggml_backend_buffer_t gbuf = vkn_internal_get_buffer(buf);
    return gbuf ? ggml_backend_vk_get_raw_buffer(gbuf) : nullptr;
}

int vkn_vk_submit_compute(void* cmd_buf, void* fence) {
    ggml_backend_t backend = vkn_internal_get_backend();
    return ggml_backend_vk_submit_compute_cmd(backend, cmd_buf, fence);
}

void vkn_vk_synchronize(void) {
    ggml_backend_t backend = vkn_internal_get_backend();
    if (backend) {
        ggml_backend_synchronize(backend);
    }
}

int vkn_vk_bincount_i32(
    const VknBuffer* x_i32, size_t x_off,
    int64_t n,
    VknBuffer* out_i32, size_t out_off,
    int64_t out_len
) {
    if (!x_i32 || !out_i32 || n < 0 || out_len <= 0 || (x_off % 4 != 0) || (out_off % 4 != 0)) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(g_direct_mutex);
    ggml_backend_t backend = vkn_internal_get_backend();
    if (!ensure_direct_locked(backend)) return -1;

    VkBuffer xb = get_vk_buffer(x_i32);
    VkBuffer ob = get_vk_buffer(out_i32);
    if (xb == VK_NULL_HANDLE || ob == VK_NULL_HANDLE) return -1;

    struct {
        uint32_t n;
        uint32_t out_len;
        uint32_t x_base;
        uint32_t out_base;
    } pc = {
        static_cast<uint32_t>(n),
        static_cast<uint32_t>(out_len),
        static_cast<uint32_t>(x_off / 4),
        static_cast<uint32_t>(out_off / 4),
    };

    return dispatch_direct(
        backend, g_state.p_bincount_i32,
        { xb, ob }, ob, out_off, static_cast<size_t>(out_len) * 4,
        &pc, sizeof(pc), n
    );
}

int vkn_vk_bincount_weighted_f32(
    const VknBuffer* x_i32, size_t x_off,
    const VknBuffer* w_f32, size_t w_off,
    int64_t n,
    VknBuffer* out_f32, size_t out_off,
    int64_t out_len
) {
    if (!x_i32 || !w_f32 || !out_f32 || n < 0 || out_len <= 0 ||
        (x_off % 4 != 0) || (w_off % 4 != 0) || (out_off % 4 != 0)) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(g_direct_mutex);
    ggml_backend_t backend = vkn_internal_get_backend();
    if (!ensure_direct_locked(backend)) return -1;

    VkBuffer xb = get_vk_buffer(x_i32);
    VkBuffer wb = get_vk_buffer(w_f32);
    VkBuffer ob = get_vk_buffer(out_f32);
    if (xb == VK_NULL_HANDLE || wb == VK_NULL_HANDLE || ob == VK_NULL_HANDLE) return -1;

    struct {
        uint32_t n;
        uint32_t out_len;
        uint32_t x_base;
        uint32_t w_base;
        uint32_t out_base;
    } pc = {
        static_cast<uint32_t>(n),
        static_cast<uint32_t>(out_len),
        static_cast<uint32_t>(x_off / 4),
        static_cast<uint32_t>(w_off / 4),
        static_cast<uint32_t>(out_off / 4),
    };

    return dispatch_direct(
        backend, g_state.p_bincount_wf32,
        { xb, wb, ob }, ob, out_off, static_cast<size_t>(out_len) * 4,
        &pc, sizeof(pc), n
    );
}

int vkn_vk_histogram2d_f32(
    const VknBuffer* x_f32, size_t x_off,
    const VknBuffer* y_f32, size_t y_off,
    int64_t n,
    float xmin, float xmax,
    float ymin, float ymax,
    int width, int height,
    int flip_y,
    VknBuffer* out_i32, size_t out_off
) {
    if (!x_f32 || !y_f32 || !out_i32 || n < 0 || width <= 0 || height <= 0 ||
        !(xmax > xmin) || !(ymax > ymin) ||
        (x_off % 4 != 0) || (y_off % 4 != 0) || (out_off % 4 != 0)) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(g_direct_mutex);
    ggml_backend_t backend = vkn_internal_get_backend();
    if (!ensure_direct_locked(backend)) return -1;

    VkBuffer xb = get_vk_buffer(x_f32);
    VkBuffer yb = get_vk_buffer(y_f32);
    VkBuffer ob = get_vk_buffer(out_i32);
    if (xb == VK_NULL_HANDLE || yb == VK_NULL_HANDLE || ob == VK_NULL_HANDLE) return -1;

    struct {
        uint32_t n;
        uint32_t width;
        uint32_t height;
        uint32_t flip_y;
        uint32_t x_base;
        uint32_t y_base;
        uint32_t out_base;
        float xmin;
        float inv_dx;
        float ymin;
        float ymax;
        float inv_dy;
    } pc = {
        static_cast<uint32_t>(n),
        static_cast<uint32_t>(width),
        static_cast<uint32_t>(height),
        static_cast<uint32_t>(flip_y ? 1u : 0u),
        static_cast<uint32_t>(x_off / 4),
        static_cast<uint32_t>(y_off / 4),
        static_cast<uint32_t>(out_off / 4),
        xmin,
        1.0f / (xmax - xmin),
        ymin,
        ymax,
        1.0f / (ymax - ymin),
    };

    size_t out_bytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 4;
    return dispatch_direct(
        backend, g_state.p_hist2d_f32,
        { xb, yb, ob }, ob, out_off, out_bytes,
        &pc, sizeof(pc), n
    );
}

int vkn_vk_histogram1d_f32(
    const VknBuffer* v_f32, size_t v_off,
    const VknBuffer* x_f32, size_t x_off,
    const VknBuffer* y_f32, size_t y_off,
    int64_t n,
    int bins,
    float vmin, float vmax,
    int has_bbox,
    float xmin, float xmax,
    float ymin, float ymax,
    VknBuffer* out_i32, size_t out_off
) {
    if (!v_f32 || !out_i32 || n < 0 || bins <= 0 || !(vmax > vmin) ||
        (v_off % 4 != 0) || (x_off % 4 != 0) || (y_off % 4 != 0) || (out_off % 4 != 0)) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(g_direct_mutex);
    ggml_backend_t backend = vkn_internal_get_backend();
    if (!ensure_direct_locked(backend)) return -1;

    VkBuffer vb = get_vk_buffer(v_f32);
    VkBuffer xb = (has_bbox && x_f32) ? get_vk_buffer(x_f32) : vb;
    VkBuffer yb = (has_bbox && y_f32) ? get_vk_buffer(y_f32) : vb;
    VkBuffer ob = get_vk_buffer(out_i32);
    if (vb == VK_NULL_HANDLE || xb == VK_NULL_HANDLE || yb == VK_NULL_HANDLE || ob == VK_NULL_HANDLE) {
        return -1;
    }

    struct {
        uint32_t n;
        uint32_t bins;
        uint32_t has_bbox;
        uint32_t v_base;
        uint32_t x_base;
        uint32_t y_base;
        uint32_t out_base;
        float    vmin;
        float    vmax;
        float    inv_dv;
        float    xmin;
        float    xmax;
        float    ymin;
        float    ymax;
    } pc = {
        static_cast<uint32_t>(n),
        static_cast<uint32_t>(bins),
        (has_bbox && x_f32 && y_f32) ? 1u : 0u,
        static_cast<uint32_t>(v_off / 4),
        static_cast<uint32_t>(x_off / 4),
        static_cast<uint32_t>(y_off / 4),
        static_cast<uint32_t>(out_off / 4),
        vmin,
        vmax,
        1.0f / (vmax - vmin),
        xmin, xmax, ymin, ymax,
    };

    return dispatch_direct(
        backend, g_state.p_hist1d_f32,
        { vb, xb, yb, ob }, ob, out_off, static_cast<size_t>(bins) * 4,
        &pc, sizeof(pc), n
    );
}

} // extern "C"
