#include "vkshader.h"

#include <vulkan/vulkan.h>

#include "vks_spv_shade_stats.h"
#include "vks_spv_shade_cdf.h"
#include "vks_spv_shade_composite.h"
#include "vks_spv_line_raster.h"
#include "vks_spv_shade_cat_composite.h"
#include "vks_spv_raster_3d.h"
#include "vks_spv_shade_3d_composite.h"
#include "vks_spv_hdr_bloom_extract.h"
#include "vks_spv_hdr_bloom_blur.h"
#include "vks_spv_hdr_bloom_composite.h"

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <vector>

namespace {

struct Pipe {
    VkShaderModule        shader_module = VK_NULL_HANDLE;
    VkDescriptorSetLayout desc_layout   = VK_NULL_HANDLE;
    VkPipelineLayout      pipe_layout   = VK_NULL_HANDLE;
    VkPipeline            pipeline      = VK_NULL_HANDLE;
    VkDescriptorSet       desc_set      = VK_NULL_HANDLE;
};

struct State {
    bool             ready        = false;
    VkDevice         dev          = VK_NULL_HANDLE;
    uint32_t         queue_family = UINT32_MAX;
    VksSubmitFn      submit_fn    = nullptr;
    VksSyncFn        sync_fn      = nullptr;
    VkCommandPool    cmd_pool     = VK_NULL_HANDLE;
    VkCommandBuffer  cmd_buf      = VK_NULL_HANDLE;
    VkFence          fence        = VK_NULL_HANDLE;
    VkDescriptorPool desc_pool    = VK_NULL_HANDLE;

    Pipe p_stats;
    Pipe p_cdf;
    Pipe p_comp;
    Pipe p_line;
    Pipe p_aux_comp;
    Pipe p_r3d;
    Pipe p_s3d;
    Pipe p_bloom_ext;
    Pipe p_bloom_blur;
    Pipe p_bloom_comp;
    VkDescriptorSet bloom_blur_sets[9] = {};
};

std::mutex g_mutex;
State g_st;

bool create_pipe(
    State& st,
    const uint32_t* spv_words,
    size_t spv_bytes,
    uint32_t num_buffers,
    uint32_t pc_bytes,
    Pipe& out
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
    pc_range.size = pc_bytes;

    VkPipelineLayoutCreateInfo pl_ci{};
    pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl_ci.setLayoutCount = 1;
    pl_ci.pSetLayouts = &out.desc_layout;
    pl_ci.pushConstantRangeCount = (pc_bytes > 0) ? 1 : 0;
    pl_ci.pPushConstantRanges = (pc_bytes > 0) ? &pc_range : nullptr;
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

void update_set(const Pipe& pipe, const std::vector<VkBuffer>& bufs) {
    std::vector<VkDescriptorBufferInfo> dbis(bufs.size());
    std::vector<VkWriteDescriptorSet> writes(bufs.size());
    for (uint32_t i = 0; i < bufs.size(); ++i) {
        dbis[i] = { bufs[i], 0, VK_WHOLE_SIZE };
        writes[i] = {};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = pipe.desc_set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &dbis[i];
    }
    vkUpdateDescriptorSets(g_st.dev, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

} // namespace

extern "C" {

int vks_init(
    void* vk_device,
    uint32_t compute_queue_family,
    VksSubmitFn submit_fn,
    VksSyncFn sync_fn
) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_st.ready) return 0;
    if (!vk_device || compute_queue_family == UINT32_MAX || !submit_fn) return -1;

    g_st.dev = static_cast<VkDevice>(vk_device);
    g_st.queue_family = compute_queue_family;
    g_st.submit_fn = submit_fn;
    g_st.sync_fn = sync_fn;

    VkCommandPoolCreateInfo cp_ci{};
    cp_ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cp_ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cp_ci.queueFamilyIndex = g_st.queue_family;
    if (vkCreateCommandPool(g_st.dev, &cp_ci, nullptr, &g_st.cmd_pool) != VK_SUCCESS) {
        return -1;
    }

    VkCommandBufferAllocateInfo cb_ai{};
    cb_ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cb_ai.commandPool = g_st.cmd_pool;
    cb_ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cb_ai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(g_st.dev, &cb_ai, &g_st.cmd_buf) != VK_SUCCESS) {
        return -1;
    }

    VkFenceCreateInfo f_ci{};
    f_ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (vkCreateFence(g_st.dev, &f_ci, nullptr, &g_st.fence) != VK_SUCCESS) {
        return -1;
    }

    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = 128;

    VkDescriptorPoolCreateInfo dp_ci{};
    dp_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dp_ci.maxSets = 32;
    dp_ci.poolSizeCount = 1;
    dp_ci.pPoolSizes = &pool_size;
    if (vkCreateDescriptorPool(g_st.dev, &dp_ci, nullptr, &g_st.desc_pool) != VK_SUCCESS) {
        return -1;
    }

    bool ok1 = create_pipe(g_st, vks_spv_shade_stats, sizeof(vks_spv_shade_stats), 2, 2 * sizeof(uint32_t), g_st.p_stats);
    bool ok2 = create_pipe(g_st, vks_spv_shade_cdf, sizeof(vks_spv_shade_cdf), 2, 0, g_st.p_cdf);
    bool ok3 = create_pipe(g_st, vks_spv_shade_composite, sizeof(vks_spv_shade_composite), 6, 15 * sizeof(uint32_t), g_st.p_comp);
    bool ok4 = create_pipe(g_st, vks_spv_line_raster, sizeof(vks_spv_line_raster), 6, 12 * sizeof(uint32_t), g_st.p_line);
    bool ok5 = create_pipe(g_st, vks_spv_shade_cat_composite, sizeof(vks_spv_shade_cat_composite), 4, 15 * sizeof(uint32_t), g_st.p_aux_comp);
    bool ok6 = create_pipe(g_st, vks_spv_raster_3d, sizeof(vks_spv_raster_3d), 8, 8 * sizeof(uint32_t), g_st.p_r3d);
    bool ok7 = create_pipe(g_st, vks_spv_shade_3d_composite, sizeof(vks_spv_shade_3d_composite), 5, 13 * sizeof(uint32_t), g_st.p_s3d);
    bool ok8 = create_pipe(g_st, vks_spv_hdr_bloom_extract, sizeof(vks_spv_hdr_bloom_extract), 3, 4 * sizeof(uint32_t), g_st.p_bloom_ext);
    bool ok9 = create_pipe(g_st, vks_spv_hdr_bloom_blur, sizeof(vks_spv_hdr_bloom_blur), 2, 5 * sizeof(uint32_t), g_st.p_bloom_blur);
    bool ok10 = create_pipe(g_st, vks_spv_hdr_bloom_composite, sizeof(vks_spv_hdr_bloom_composite), 5, 10 * sizeof(uint32_t), g_st.p_bloom_comp);

    bool ok11 = true;
    if (ok9) {
        VkDescriptorSetLayout layouts[9];
        for (int i = 0; i < 9; ++i) layouts[i] = g_st.p_bloom_blur.desc_layout;
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = g_st.desc_pool;
        ai.descriptorSetCount = 9;
        ai.pSetLayouts = layouts;
        ok11 = (vkAllocateDescriptorSets(g_st.dev, &ai, g_st.bloom_blur_sets) == VK_SUCCESS);
    }

    g_st.ready = ok1 && ok2 && ok3 && ok4 && ok5 && ok6 && ok7 && ok8 && ok9 && ok10 && ok11;
    return g_st.ready ? 0 : -1;
}

int vks_shade_rgba(
    void* grid_vkbuf, size_t grid_off,
    int width, int height,
    int how, float alpha,
    int spread_px,
    void* bg_vkbuf, size_t bg_off,
    int bg_w, int bg_h,
    float uv_left, float uv_top, float uv_right, float uv_bottom,
    void* stats_vkbuf,
    void* cdf_vkbuf,
    void* lut_vkbuf,
    void* out_vkbuf, size_t out_off
) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_st.ready || !grid_vkbuf || !stats_vkbuf || !cdf_vkbuf || !lut_vkbuf || !out_vkbuf) return -1;
    if (width <= 0 || height <= 0 || (grid_off % 4 != 0) || (bg_off % 4 != 0) || (out_off % 4 != 0)) {
        return -1;
    }

    VkBuffer gb  = static_cast<VkBuffer>(grid_vkbuf);
    VkBuffer sb  = static_cast<VkBuffer>(stats_vkbuf);
    VkBuffer cb  = static_cast<VkBuffer>(cdf_vkbuf);
    VkBuffer lb  = static_cast<VkBuffer>(lut_vkbuf);
    VkBuffer ob  = static_cast<VkBuffer>(out_vkbuf);
    VkBuffer bgb = (bg_vkbuf && bg_w > 0 && bg_h > 0) ? static_cast<VkBuffer>(bg_vkbuf) : sb;

    if (g_st.sync_fn) {
        g_st.sync_fn();
    }

    update_set(g_st.p_stats, { gb, sb });
    update_set(g_st.p_cdf,   { sb, cb });
    update_set(g_st.p_comp,  { gb, sb, cb, bgb, ob, lb });

    VkCommandBuffer cmd = g_st.cmd_buf;
    if (vkResetCommandBuffer(cmd, 0) != VK_SUCCESS) return -1;

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS) return -1;

    uint32_t total = static_cast<uint32_t>(width) * static_cast<uint32_t>(height);
    uint32_t groups = std::min<uint32_t>((total + 255u) / 256u, 32768u);

    vkCmdFillBuffer(cmd, sb, 0, 2050 * sizeof(uint32_t), 0);

    VkMemoryBarrier bar_xfer{};
    bar_xfer.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    bar_xfer.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bar_xfer.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &bar_xfer, 0, nullptr, 0, nullptr
    );

    struct {
        uint32_t n;
        uint32_t grid_base;
    } pc_stats = { total, static_cast<uint32_t>(grid_off / 4) };

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_st.p_stats.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_st.p_stats.pipe_layout, 0, 1, &g_st.p_stats.desc_set, 0, nullptr);
    vkCmdPushConstants(cmd, g_st.p_stats.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_stats), &pc_stats);
    vkCmdDispatch(cmd, groups, 1, 1);

    VkMemoryBarrier bar_comp{};
    bar_comp.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    bar_comp.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bar_comp.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &bar_comp, 0, nullptr, 0, nullptr
    );

    if (how == 0) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_st.p_cdf.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_st.p_cdf.pipe_layout, 0, 1, &g_st.p_cdf.desc_set, 0, nullptr);
        vkCmdDispatch(cmd, 1, 1, 1);

        vkCmdPipelineBarrier(
            cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &bar_comp, 0, nullptr, 0, nullptr
        );
    }

    struct {
        uint32_t width;
        uint32_t height;
        uint32_t how;
        uint32_t has_bg;
        uint32_t bg_w;
        uint32_t bg_h;
        uint32_t grid_base;
        uint32_t bg_base;
        uint32_t out_base;
        uint32_t spread_px;
        float    alpha;
        float    uv_left;
        float    uv_top;
        float    uv_right;
        float    uv_bottom;
    } pc_comp = {
        static_cast<uint32_t>(width),
        static_cast<uint32_t>(height),
        static_cast<uint32_t>(how),
        (bg_vkbuf && bg_w > 0 && bg_h > 0) ? 1u : 0u,
        static_cast<uint32_t>(bg_w > 0 ? bg_w : 0),
        static_cast<uint32_t>(bg_h > 0 ? bg_h : 0),
        static_cast<uint32_t>(grid_off / 4),
        static_cast<uint32_t>(bg_off / 4),
        static_cast<uint32_t>(out_off / 4),
        (spread_px < 0) ? 0xFFFFFFFFu : static_cast<uint32_t>(spread_px),
        alpha,
        uv_left, uv_top, uv_right, uv_bottom,
    };

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_st.p_comp.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_st.p_comp.pipe_layout, 0, 1, &g_st.p_comp.desc_set, 0, nullptr);
    vkCmdPushConstants(cmd, g_st.p_comp.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_comp), &pc_comp);
    vkCmdDispatch(cmd, groups, 1, 1);

    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) return -1;

    if (g_st.submit_fn(cmd, g_st.fence) != 0) {
        return -1;
    }
    VkResult wrc = vkWaitForFences(g_st.dev, 1, &g_st.fence, VK_TRUE, UINT64_MAX);
    VkResult rrc = vkResetFences(g_st.dev, 1, &g_st.fence);
    return (wrc == VK_SUCCESS && rrc == VK_SUCCESS) ? 0 : -1;
}

int vks_raster_lines(
    void* x_vkbuf, size_t x_off,
    void* y_vkbuf, size_t y_off,
    void* vals_vkbuf, size_t vals_off,
    void* palette_vkbuf,
    uint32_t n,
    int width, int height,
    int mode,
    float xmin, float xmax, float ymin, float ymax,
    void* count_vkbuf,
    void* aux_vkbuf
) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_st.ready || !x_vkbuf || !y_vkbuf || !count_vkbuf) return -1;
    if (width <= 0 || height <= 0 || (x_off % 4 != 0) || (y_off % 4 != 0) || (vals_off % 4 != 0)) {
        return -1;
    }

    VkBuffer xb  = static_cast<VkBuffer>(x_vkbuf);
    VkBuffer yb  = static_cast<VkBuffer>(y_vkbuf);
    VkBuffer vb  = vals_vkbuf ? static_cast<VkBuffer>(vals_vkbuf) : xb;
    VkBuffer pb  = palette_vkbuf ? static_cast<VkBuffer>(palette_vkbuf) : xb;
    VkBuffer cb  = static_cast<VkBuffer>(count_vkbuf);
    VkBuffer ab  = aux_vkbuf ? static_cast<VkBuffer>(aux_vkbuf) : cb;

    if (g_st.sync_fn) {
        g_st.sync_fn();
    }

    update_set(g_st.p_line, { xb, yb, vb, pb, cb, ab });

    VkCommandBuffer cmd = g_st.cmd_buf;
    if (vkResetCommandBuffer(cmd, 0) != VK_SUCCESS) return -1;

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS) return -1;

    uint64_t total_px = static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
    vkCmdFillBuffer(cmd, cb, 0, total_px * sizeof(uint32_t), 0);
    if (mode == 1 && aux_vkbuf) {
        vkCmdFillBuffer(cmd, ab, 0, total_px * sizeof(uint32_t), 0);
    } else if (mode == 2 && aux_vkbuf) {
        vkCmdFillBuffer(cmd, ab, 0, total_px * 3u * sizeof(uint32_t), 0);
    }

    VkMemoryBarrier bar_xfer{};
    bar_xfer.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    bar_xfer.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bar_xfer.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &bar_xfer, 0, nullptr, 0, nullptr
    );

    if (n >= 2u) {
        struct {
            uint32_t n;
            uint32_t width;
            uint32_t height;
            uint32_t mode;
            uint32_t x_base;
            uint32_t y_base;
            uint32_t v_base;
            uint32_t pad;
            float    xmin;
            float    xmax;
            float    ymin;
            float    ymax;
        } pc = {
            n,
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height),
            static_cast<uint32_t>(mode),
            static_cast<uint32_t>(x_off / 4),
            static_cast<uint32_t>(y_off / 4),
            static_cast<uint32_t>(vals_off / 4),
            0u,
            xmin, xmax, ymin, ymax,
        };

        uint32_t groups = std::min<uint32_t>(((n - 1u) + 255u) / 256u, 32768u);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_st.p_line.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_st.p_line.pipe_layout, 0, 1, &g_st.p_line.desc_set, 0, nullptr);
        vkCmdPushConstants(cmd, g_st.p_line.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, groups, 1, 1);
    }

    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) return -1;
    if (g_st.submit_fn(cmd, g_st.fence) != 0) return -1;
    VkResult wrc = vkWaitForFences(g_st.dev, 1, &g_st.fence, VK_TRUE, UINT64_MAX);
    VkResult rrc = vkResetFences(g_st.dev, 1, &g_st.fence);
    return (wrc == VK_SUCCESS && rrc == VK_SUCCESS) ? 0 : -1;
}

int vks_shade_aux_rgba(
    void* count_vkbuf,
    void* aux_vkbuf,
    int width, int height,
    int mode, float alpha,
    float vmin, float vmax,
    void* bg_vkbuf, size_t bg_off,
    int bg_w, int bg_h,
    float uv_left, float uv_top, float uv_right, float uv_bottom,
    void* out_vkbuf, size_t out_off
) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_st.ready || !count_vkbuf || !aux_vkbuf || !out_vkbuf) return -1;
    if (width <= 0 || height <= 0 || (bg_off % 4 != 0) || (out_off % 4 != 0)) return -1;

    VkBuffer cb  = static_cast<VkBuffer>(count_vkbuf);
    VkBuffer ab  = static_cast<VkBuffer>(aux_vkbuf);
    VkBuffer ob  = static_cast<VkBuffer>(out_vkbuf);
    VkBuffer bgb = (bg_vkbuf && bg_w > 0 && bg_h > 0) ? static_cast<VkBuffer>(bg_vkbuf) : cb;

    if (g_st.sync_fn) {
        g_st.sync_fn();
    }

    update_set(g_st.p_aux_comp, { cb, ab, bgb, ob });

    VkCommandBuffer cmd = g_st.cmd_buf;
    if (vkResetCommandBuffer(cmd, 0) != VK_SUCCESS) return -1;

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS) return -1;

    struct {
        uint32_t width;
        uint32_t height;
        uint32_t mode;
        uint32_t has_bg;
        uint32_t bg_w;
        uint32_t bg_h;
        uint32_t bg_base;
        uint32_t out_base;
        float    alpha;
        float    vmin;
        float    vmax;
        float    uv_left;
        float    uv_top;
        float    uv_right;
        float    uv_bottom;
    } pc = {
        static_cast<uint32_t>(width),
        static_cast<uint32_t>(height),
        static_cast<uint32_t>(mode),
        (bg_vkbuf && bg_w > 0 && bg_h > 0) ? 1u : 0u,
        static_cast<uint32_t>(bg_w > 0 ? bg_w : 0),
        static_cast<uint32_t>(bg_h > 0 ? bg_h : 0),
        static_cast<uint32_t>(bg_off / 4),
        static_cast<uint32_t>(out_off / 4),
        alpha,
        vmin,
        vmax,
        uv_left, uv_top, uv_right, uv_bottom,
    };

    uint32_t total = static_cast<uint32_t>(width) * static_cast<uint32_t>(height);
    uint32_t groups = std::min<uint32_t>((total + 255u) / 256u, 32768u);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_st.p_aux_comp.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_st.p_aux_comp.pipe_layout, 0, 1, &g_st.p_aux_comp.desc_set, 0, nullptr);
    vkCmdPushConstants(cmd, g_st.p_aux_comp.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, groups, 1, 1);

    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) return -1;
    if (g_st.submit_fn(cmd, g_st.fence) != 0) return -1;
    VkResult wrc = vkWaitForFences(g_st.dev, 1, &g_st.fence, VK_TRUE, UINT64_MAX);
    VkResult rrc = vkResetFences(g_st.dev, 1, &g_st.fence);
    return (wrc == VK_SUCCESS && rrc == VK_SUCCESS) ? 0 : -1;
}

int vks_raster_3d(
    void* x_vkbuf, size_t x_off,
    void* y_vkbuf, size_t y_off,
    void* z_vkbuf, size_t z_off,
    void* vals_vkbuf, size_t vals_off,
    void* palette_vkbuf,
    void* cam_vkbuf,
    uint32_t n,
    int width, int height,
    int mode,
    void* count_vkbuf,
    void* aux_vkbuf
) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_st.ready || !x_vkbuf || !y_vkbuf || !z_vkbuf || !cam_vkbuf || !count_vkbuf || !aux_vkbuf) return -1;
    if (width <= 0 || height <= 0 || (x_off % 4 != 0) || (y_off % 4 != 0) || (z_off % 4 != 0) || (vals_off % 4 != 0)) {
        return -1;
    }

    VkBuffer xb   = static_cast<VkBuffer>(x_vkbuf);
    VkBuffer yb   = static_cast<VkBuffer>(y_vkbuf);
    VkBuffer zb   = static_cast<VkBuffer>(z_vkbuf);
    VkBuffer vb   = vals_vkbuf ? static_cast<VkBuffer>(vals_vkbuf) : xb;
    VkBuffer pb   = palette_vkbuf ? static_cast<VkBuffer>(palette_vkbuf) : xb;
    VkBuffer cb   = static_cast<VkBuffer>(count_vkbuf);
    VkBuffer ab   = static_cast<VkBuffer>(aux_vkbuf);
    VkBuffer camb = static_cast<VkBuffer>(cam_vkbuf);

    if (g_st.sync_fn) {
        g_st.sync_fn();
    }

    update_set(g_st.p_r3d, { xb, yb, zb, vb, pb, cb, ab, camb });

    VkCommandBuffer cmd = g_st.cmd_buf;
    if (vkResetCommandBuffer(cmd, 0) != VK_SUCCESS) return -1;

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS) return -1;

    uint64_t total_px = static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
    vkCmdFillBuffer(cmd, cb, 0, total_px * sizeof(uint32_t), (mode == 4) ? 0x7FFFFFFFu : 0u);
    if (mode != 4) {
        vkCmdFillBuffer(cmd, ab, 0, total_px * 3u * sizeof(uint32_t), 0);
    }

    VkMemoryBarrier bar_xfer{};
    bar_xfer.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    bar_xfer.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bar_xfer.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &bar_xfer, 0, nullptr, 0, nullptr
    );

    if (n > 0u) {
        struct {
            uint32_t n;
            uint32_t width;
            uint32_t height;
            uint32_t mode;
            uint32_t x_base;
            uint32_t y_base;
            uint32_t z_base;
            uint32_t v_base;
        } pc = {
            n,
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height),
            static_cast<uint32_t>(mode),
            static_cast<uint32_t>(x_off / 4),
            static_cast<uint32_t>(y_off / 4),
            static_cast<uint32_t>(z_off / 4),
            static_cast<uint32_t>(vals_off / 4),
        };

        uint32_t groups = std::min<uint32_t>((n + 255u) / 256u, 32768u);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_st.p_r3d.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_st.p_r3d.pipe_layout, 0, 1, &g_st.p_r3d.desc_set, 0, nullptr);
        vkCmdPushConstants(cmd, g_st.p_r3d.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, groups, 1, 1);
    }

    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) return -1;
    if (g_st.submit_fn(cmd, g_st.fence) != 0) return -1;
    VkResult wrc = vkWaitForFences(g_st.dev, 1, &g_st.fence, VK_TRUE, UINT64_MAX);
    VkResult rrc = vkResetFences(g_st.dev, 1, &g_st.fence);
    return (wrc == VK_SUCCESS && rrc == VK_SUCCESS) ? 0 : -1;
}

int vks_shade_3d_rgba(
    void* count_vkbuf,
    void* aux_vkbuf,
    void* cam_vkbuf,
    int width, int height,
    float alpha, float max_count,
    void* bg_vkbuf, size_t bg_off,
    int bg_w, int bg_h,
    float uv_left, float uv_top, float uv_right, float uv_bottom,
    void* out_vkbuf, size_t out_off
) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_st.ready || !count_vkbuf || !aux_vkbuf || !cam_vkbuf || !out_vkbuf) return -1;
    if (width <= 0 || height <= 0 || (bg_off % 4 != 0) || (out_off % 4 != 0)) return -1;

    VkBuffer cb   = static_cast<VkBuffer>(count_vkbuf);
    VkBuffer ab   = static_cast<VkBuffer>(aux_vkbuf);
    VkBuffer camb = static_cast<VkBuffer>(cam_vkbuf);
    VkBuffer ob   = static_cast<VkBuffer>(out_vkbuf);
    VkBuffer bgb  = (bg_vkbuf && bg_w > 0 && bg_h > 0) ? static_cast<VkBuffer>(bg_vkbuf) : cb;

    if (g_st.sync_fn) {
        g_st.sync_fn();
    }

    update_set(g_st.p_s3d, { cb, ab, bgb, camb, ob });

    VkCommandBuffer cmd = g_st.cmd_buf;
    if (vkResetCommandBuffer(cmd, 0) != VK_SUCCESS) return -1;

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS) return -1;

    struct {
        uint32_t width;
        uint32_t height;
        uint32_t has_bg;
        uint32_t bg_w;
        uint32_t bg_h;
        uint32_t bg_base;
        uint32_t out_base;
        float    alpha;
        float    max_count;
        float    uv_left;
        float    uv_top;
        float    uv_right;
        float    uv_bottom;
    } pc = {
        static_cast<uint32_t>(width),
        static_cast<uint32_t>(height),
        (bg_vkbuf && bg_w > 0 && bg_h > 0) ? 1u : 0u,
        static_cast<uint32_t>(bg_w > 0 ? bg_w : 0),
        static_cast<uint32_t>(bg_h > 0 ? bg_h : 0),
        static_cast<uint32_t>(bg_off / 4),
        static_cast<uint32_t>(out_off / 4),
        alpha,
        max_count,
        uv_left, uv_top, uv_right, uv_bottom,
    };

    uint32_t total = static_cast<uint32_t>(width) * static_cast<uint32_t>(height);
    uint32_t groups = std::min<uint32_t>((total + 255u) / 256u, 32768u);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_st.p_s3d.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_st.p_s3d.pipe_layout, 0, 1, &g_st.p_s3d.desc_set, 0, nullptr);
    vkCmdPushConstants(cmd, g_st.p_s3d.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, groups, 1, 1);

    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) return -1;
    if (g_st.submit_fn(cmd, g_st.fence) != 0) return -1;
    VkResult wrc = vkWaitForFences(g_st.dev, 1, &g_st.fence, VK_TRUE, UINT64_MAX);
    VkResult rrc = vkResetFences(g_st.dev, 1, &g_st.fence);
    return (wrc == VK_SUCCESS && rrc == VK_SUCCESS) ? 0 : -1;
}

int vks_hdr_bloom_rgba(
    void* in_rgba_vkbuf,
    void* out_rgba_vkbuf,
    void* hdr_vkbuf,
    void* bright_vkbuf,
    void* mip1_vkbuf,
    void* mip2_vkbuf,
    void* mip3_vkbuf,
    void* tmp_vkbuf,
    int width, int height,
    float intensity,
    float exposure,
    float threshold,
    float hdr_lift
) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_st.ready || !in_rgba_vkbuf || !out_rgba_vkbuf || !hdr_vkbuf || !bright_vkbuf ||
        !mip1_vkbuf || !mip2_vkbuf || !mip3_vkbuf || !tmp_vkbuf || width <= 0 || height <= 0) {
        return -1;
    }

    VkBuffer in_b   = static_cast<VkBuffer>(in_rgba_vkbuf);
    VkBuffer out_b  = static_cast<VkBuffer>(out_rgba_vkbuf);
    VkBuffer hdr_b  = static_cast<VkBuffer>(hdr_vkbuf);
    VkBuffer br_b   = static_cast<VkBuffer>(bright_vkbuf);
    VkBuffer m1_b   = static_cast<VkBuffer>(mip1_vkbuf);
    VkBuffer m2_b   = static_cast<VkBuffer>(mip2_vkbuf);
    VkBuffer m3_b   = static_cast<VkBuffer>(mip3_vkbuf);
    VkBuffer tmp_b  = static_cast<VkBuffer>(tmp_vkbuf);

    if (g_st.sync_fn) g_st.sync_fn();

    auto update_raw_set = [](VkDescriptorSet ds, VkBuffer b0, VkBuffer b1) {
        VkDescriptorBufferInfo dbis[2] = { { b0, 0, VK_WHOLE_SIZE }, { b1, 0, VK_WHOLE_SIZE } };
        VkWriteDescriptorSet ws[2] = {};
        for (int i = 0; i < 2; ++i) {
            ws[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            ws[i].dstSet = ds;
            ws[i].dstBinding = static_cast<uint32_t>(i);
            ws[i].descriptorCount = 1;
            ws[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            ws[i].pBufferInfo = &dbis[i];
        }
        vkUpdateDescriptorSets(g_st.dev, 2, ws, 0, nullptr);
    };

    update_set(g_st.p_bloom_ext, { in_b, hdr_b, br_b });
    update_raw_set(g_st.bloom_blur_sets[0], br_b,  m1_b);
    update_raw_set(g_st.bloom_blur_sets[1], m1_b,  tmp_b);
    update_raw_set(g_st.bloom_blur_sets[2], tmp_b, m1_b);
    update_raw_set(g_st.bloom_blur_sets[3], m1_b,  m2_b);
    update_raw_set(g_st.bloom_blur_sets[4], m2_b,  tmp_b);
    update_raw_set(g_st.bloom_blur_sets[5], tmp_b, m2_b);
    update_raw_set(g_st.bloom_blur_sets[6], m2_b,  m3_b);
    update_raw_set(g_st.bloom_blur_sets[7], m3_b,  tmp_b);
    update_raw_set(g_st.bloom_blur_sets[8], tmp_b, m3_b);
    update_set(g_st.p_bloom_comp, { hdr_b, m1_b, m2_b, m3_b, out_b });

    int w1 = std::max(1, width / 2),  h1 = std::max(1, height / 2);
    int w2 = std::max(1, w1 / 2),     h2 = std::max(1, h1 / 2);
    int w3 = std::max(1, w2 / 2),     h3 = std::max(1, h2 / 2);

    VkCommandBuffer cmd = g_st.cmd_buf;
    if (vkResetCommandBuffer(cmd, 0) != VK_SUCCESS) return -1;
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS) return -1;

    VkMemoryBarrier bar{};
    bar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

    auto gx = [](int w) { return static_cast<uint32_t>((w + 15) / 16); };
    auto gy = [](int h) { return static_cast<uint32_t>((h + 15) / 16); };

    // Pass 1: Extract linear HDR + soft-knee bright pass
    struct { int w; int h; float th; float lift; } pc_ext = { width, height, threshold, hdr_lift };
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_st.p_bloom_ext.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_st.p_bloom_ext.pipe_layout, 0, 1, &g_st.p_bloom_ext.desc_set, 0, nullptr);
    vkCmdPushConstants(cmd, g_st.p_bloom_ext.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_ext), &pc_ext);
    vkCmdDispatch(cmd, gx(width), gy(height), 1);
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &bar, 0, nullptr, 0, nullptr);

    // Passes 2..10: 3-mip downsample + separable 9-tap Gaussian blur pyramid
    struct BlurStep { int set_idx; int sw; int sh; int dw; int dh; int mode; };
    const BlurStep steps[9] = {
        { 0, width, height, w1, h1, 0 },
        { 1, w1,    h1,     w1, h1, 1 },
        { 2, w1,    h1,     w1, h1, 2 },
        { 3, w1,    h1,     w2, h2, 0 },
        { 4, w2,    h2,     w2, h2, 1 },
        { 5, w2,    h2,     w2, h2, 2 },
        { 6, w2,    h2,     w3, h3, 0 },
        { 7, w3,    h3,     w3, h3, 1 },
        { 8, w3,    h3,     w3, h3, 2 },
    };
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_st.p_bloom_blur.pipeline);
    for (int s = 0; s < 9; ++s) {
        const auto& st = steps[s];
        struct { int sw; int sh; int dw; int dh; int mode; } pc_b = { st.sw, st.sh, st.dw, st.dh, st.mode };
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_st.p_bloom_blur.pipe_layout, 0, 1, &g_st.bloom_blur_sets[st.set_idx], 0, nullptr);
        vkCmdPushConstants(cmd, g_st.p_bloom_blur.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_b), &pc_b);
        vkCmdDispatch(cmd, gx(st.dw), gy(st.dh), 1);
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &bar, 0, nullptr, 0, nullptr);
    }

    // Pass 11: Bilinear multi-mip composite + ACES Filmic tone mapping
    struct { int w; int h; int w1; int h1; int w2; int h2; int w3; int h3; float intensity; float exposure; } pc_c = {
        width, height, w1, h1, w2, h2, w3, h3, intensity, exposure
    };
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_st.p_bloom_comp.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_st.p_bloom_comp.pipe_layout, 0, 1, &g_st.p_bloom_comp.desc_set, 0, nullptr);
    vkCmdPushConstants(cmd, g_st.p_bloom_comp.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_c), &pc_c);
    vkCmdDispatch(cmd, gx(width), gy(height), 1);

    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) return -1;
    if (g_st.submit_fn(cmd, g_st.fence) != 0) return -1;
    VkResult wrc = vkWaitForFences(g_st.dev, 1, &g_st.fence, VK_TRUE, UINT64_MAX);
    VkResult rrc = vkResetFences(g_st.dev, 1, &g_st.fence);
    return (wrc == VK_SUCCESS && rrc == VK_SUCCESS) ? 0 : -1;
}

} // extern "C"
