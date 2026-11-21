#include "vkshader.h"

#include <vulkan/vulkan.h>

#include "vks_spv_shade_stats.h"
#include "vks_spv_shade_cdf.h"
#include "vks_spv_shade_composite.h"

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
    pool_size.descriptorCount = 16;

    VkDescriptorPoolCreateInfo dp_ci{};
    dp_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dp_ci.maxSets = 4;
    dp_ci.poolSizeCount = 1;
    dp_ci.pPoolSizes = &pool_size;
    if (vkCreateDescriptorPool(g_st.dev, &dp_ci, nullptr, &g_st.desc_pool) != VK_SUCCESS) {
        return -1;
    }

    bool ok1 = create_pipe(g_st, vks_spv_shade_stats, sizeof(vks_spv_shade_stats), 2, 2 * sizeof(uint32_t), g_st.p_stats);
    bool ok2 = create_pipe(g_st, vks_spv_shade_cdf, sizeof(vks_spv_shade_cdf), 2, 0, g_st.p_cdf);
    bool ok3 = create_pipe(g_st, vks_spv_shade_composite, sizeof(vks_spv_shade_composite), 5, 15 * sizeof(uint32_t), g_st.p_comp);

    g_st.ready = ok1 && ok2 && ok3;
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
    void* out_vkbuf, size_t out_off
) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_st.ready || !grid_vkbuf || !stats_vkbuf || !cdf_vkbuf || !out_vkbuf) return -1;
    if (width <= 0 || height <= 0 || (grid_off % 4 != 0) || (bg_off % 4 != 0) || (out_off % 4 != 0)) {
        return -1;
    }

    VkBuffer gb  = static_cast<VkBuffer>(grid_vkbuf);
    VkBuffer sb  = static_cast<VkBuffer>(stats_vkbuf);
    VkBuffer cb  = static_cast<VkBuffer>(cdf_vkbuf);
    VkBuffer ob  = static_cast<VkBuffer>(out_vkbuf);
    VkBuffer bgb = (bg_vkbuf && bg_w > 0 && bg_h > 0) ? static_cast<VkBuffer>(bg_vkbuf) : sb;

    if (g_st.sync_fn) {
        g_st.sync_fn();
    }

    update_set(g_st.p_stats, { gb, sb });
    update_set(g_st.p_cdf,   { sb, cb });
    update_set(g_st.p_comp,  { gb, sb, cb, bgb, ob });

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

} // extern "C"
