#include "vkspatial.h"

#include <vulkan/vulkan.h>

#include "vksp_spv_morton_2d.h"
#include "vksp_spv_point_in_polygon.h"
#include "vksp_spv_sjoin_polygons.h"
#include "vksp_spv_bh_bin.h"
#include "vksp_spv_bh_leaves.h"
#include "vksp_spv_bh_reduce.h"
#include "vksp_spv_bh_step.h"

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
    VkspSubmitFn     submit_fn    = nullptr;
    VkspSyncFn       sync_fn      = nullptr;
    VkCommandPool    cmd_pool     = VK_NULL_HANDLE;
    VkCommandBuffer  cmd_buf      = VK_NULL_HANDLE;
    VkFence          fence        = VK_NULL_HANDLE;
    VkDescriptorPool desc_pool    = VK_NULL_HANDLE;

    Pipe p_morton;
    Pipe p_pip;
    Pipe p_sjoin;
    Pipe p_bh_bin;
    Pipe p_bh_leaves;
    Pipe p_bh_reduce;
    Pipe p_bh_step;
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
    if (vkCreateShaderModule(st.dev, &sm_ci, nullptr, &out.shader_module) != VK_SUCCESS) return false;

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
    if (vkCreateDescriptorSetLayout(st.dev, &dsl_ci, nullptr, &out.desc_layout) != VK_SUCCESS) return false;

    VkPushConstantRange pc_range{};
    pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc_range.offset = 0;
    pc_range.size = pc_bytes;

    VkPipelineLayoutCreateInfo pl_ci{};
    pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl_ci.setLayoutCount = 1;
    pl_ci.pSetLayouts = &out.desc_layout;
    pl_ci.pushConstantRangeCount = 1;
    pl_ci.pPushConstantRanges = &pc_range;
    if (vkCreatePipelineLayout(st.dev, &pl_ci, nullptr, &out.pipe_layout) != VK_SUCCESS) return false;

    VkComputePipelineCreateInfo cp_ci{};
    cp_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cp_ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cp_ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cp_ci.stage.module = out.shader_module;
    cp_ci.stage.pName = "main";
    cp_ci.layout = out.pipe_layout;
    if (vkCreateComputePipelines(st.dev, VK_NULL_HANDLE, 1, &cp_ci, nullptr, &out.pipeline) != VK_SUCCESS) return false;

    VkDescriptorSetAllocateInfo dsa_ci{};
    dsa_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsa_ci.descriptorPool = st.desc_pool;
    dsa_ci.descriptorSetCount = 1;
    dsa_ci.pSetLayouts = &out.desc_layout;
    return vkAllocateDescriptorSets(st.dev, &dsa_ci, &out.desc_set) == VK_SUCCESS;
}

void update_set(const Pipe& pipe, const std::vector<VkBuffer>& bufs) {
    std::vector<VkDescriptorBufferInfo> dbis(bufs.size());
    std::vector<VkWriteDescriptorSet> writes(bufs.size());
    for (size_t i = 0; i < bufs.size(); ++i) {
        dbis[i] = { bufs[i], 0, VK_WHOLE_SIZE };
        writes[i] = {};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = pipe.desc_set;
        writes[i].dstBinding = static_cast<uint32_t>(i);
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &dbis[i];
    }
    vkUpdateDescriptorSets(g_st.dev, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

int dispatch_pipe(
    const Pipe& pipe,
    const std::vector<VkBuffer>& bufs,
    const void* pc_data,
    uint32_t pc_bytes,
    int64_t n
) {
    if (g_st.sync_fn) g_st.sync_fn();

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

    VkCommandBuffer cmd = g_st.cmd_buf;
    if (vkResetCommandBuffer(cmd, 0) != VK_SUCCESS) return -1;

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS) return -1;

    if (n > 0) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipe_layout, 0, 1, &pipe.desc_set, 0, nullptr);
        vkCmdPushConstants(cmd, pipe.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pc_bytes, pc_data);
        uint32_t groups = static_cast<uint32_t>(std::min<int64_t>((n + 255) / 256, 32768));
        vkCmdDispatch(cmd, groups, 1, 1);
    }

    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) return -1;
    if (g_st.submit_fn(cmd, g_st.fence) != 0) return -1;

    VkResult wrc = vkWaitForFences(g_st.dev, 1, &g_st.fence, VK_TRUE, UINT64_MAX);
    VkResult rrc = vkResetFences(g_st.dev, 1, &g_st.fence);
    return (wrc == VK_SUCCESS && rrc == VK_SUCCESS) ? 0 : -1;
}

} // namespace

extern "C" {

int vksp_init(
    void* vk_device,
    uint32_t compute_queue_family,
    VkspSubmitFn submit_fn,
    VkspSyncFn sync_fn
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
    if (vkCreateCommandPool(g_st.dev, &cp_ci, nullptr, &g_st.cmd_pool) != VK_SUCCESS) return -1;

    VkCommandBufferAllocateInfo cb_ai{};
    cb_ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cb_ai.commandPool = g_st.cmd_pool;
    cb_ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cb_ai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(g_st.dev, &cb_ai, &g_st.cmd_buf) != VK_SUCCESS) return -1;

    VkFenceCreateInfo f_ci{};
    f_ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (vkCreateFence(g_st.dev, &f_ci, nullptr, &g_st.fence) != VK_SUCCESS) return -1;

    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = 64;

    VkDescriptorPoolCreateInfo dp_ci{};
    dp_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dp_ci.maxSets = 16;
    dp_ci.poolSizeCount = 1;
    dp_ci.pPoolSizes = &pool_size;
    if (vkCreateDescriptorPool(g_st.dev, &dp_ci, nullptr, &g_st.desc_pool) != VK_SUCCESS) return -1;

    bool ok1 = create_pipe(g_st, vksp_spv_morton_2d, sizeof(vksp_spv_morton_2d), 3, 8 * sizeof(uint32_t), g_st.p_morton);
    bool ok2 = create_pipe(g_st, vksp_spv_point_in_polygon, sizeof(vksp_spv_point_in_polygon), 4, 6 * sizeof(uint32_t), g_st.p_pip);
    bool ok3 = create_pipe(g_st, vksp_spv_sjoin_polygons, sizeof(vksp_spv_sjoin_polygons), 6, 5 * sizeof(uint32_t), g_st.p_sjoin);
    bool ok4 = create_pipe(g_st, vksp_spv_bh_bin, sizeof(vksp_spv_bh_bin), 3, 2 * sizeof(uint32_t), g_st.p_bh_bin);
    bool ok5 = create_pipe(g_st, vksp_spv_bh_leaves, sizeof(vksp_spv_bh_leaves), 5, 3 * sizeof(uint32_t), g_st.p_bh_leaves);
    bool ok6 = create_pipe(g_st, vksp_spv_bh_reduce, sizeof(vksp_spv_bh_reduce), 2, 2 * sizeof(uint32_t), g_st.p_bh_reduce);
    bool ok7 = create_pipe(g_st, vksp_spv_bh_step, sizeof(vksp_spv_bh_step), 8, 9 * sizeof(uint32_t), g_st.p_bh_step);

    g_st.ready = ok1 && ok2 && ok3 && ok4 && ok5 && ok6 && ok7;
    return g_st.ready ? 0 : -1;
}

int vksp_morton_2d(
    void* x_vkbuf, size_t x_off,
    void* y_vkbuf, size_t y_off,
    int64_t n,
    float xmin, float xmax,
    float ymin, float ymax,
    void* out_vkbuf, size_t out_off
) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_st.ready || !x_vkbuf || !y_vkbuf || !out_vkbuf || n < 0 ||
        !(xmax > xmin) || !(ymax > ymin) || (x_off % 4 != 0) || (y_off % 4 != 0) || (out_off % 4 != 0)) {
        return -1;
    }

    struct {
        uint32_t n;
        uint32_t x_base;
        uint32_t y_base;
        uint32_t out_base;
        float    xmin;
        float    inv_dx;
        float    ymin;
        float    inv_dy;
    } pc = {
        static_cast<uint32_t>(n),
        static_cast<uint32_t>(x_off / 4),
        static_cast<uint32_t>(y_off / 4),
        static_cast<uint32_t>(out_off / 4),
        xmin, 1.0f / (xmax - xmin),
        ymin, 1.0f / (ymax - ymin),
    };

    return dispatch_pipe(
        g_st.p_morton,
        { static_cast<VkBuffer>(x_vkbuf), static_cast<VkBuffer>(y_vkbuf), static_cast<VkBuffer>(out_vkbuf) },
        &pc, sizeof(pc), n
    );
}

int vksp_point_in_polygon(
    void* x_vkbuf, size_t x_off,
    void* y_vkbuf, size_t y_off,
    int64_t n,
    void* poly_vkbuf, size_t poly_off,
    int n_verts,
    void* out_vkbuf, size_t out_off
) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_st.ready || !x_vkbuf || !y_vkbuf || !poly_vkbuf || !out_vkbuf || n < 0 || n_verts < 3 ||
        (x_off % 4 != 0) || (y_off % 4 != 0) || (poly_off % 4 != 0) || (out_off % 4 != 0)) {
        return -1;
    }

    struct {
        uint32_t n;
        uint32_t n_verts;
        uint32_t x_base;
        uint32_t y_base;
        uint32_t poly_base;
        uint32_t out_base;
    } pc = {
        static_cast<uint32_t>(n),
        static_cast<uint32_t>(n_verts),
        static_cast<uint32_t>(x_off / 4),
        static_cast<uint32_t>(y_off / 4),
        static_cast<uint32_t>(poly_off / 4),
        static_cast<uint32_t>(out_off / 4),
    };

    return dispatch_pipe(
        g_st.p_pip,
        {
            static_cast<VkBuffer>(x_vkbuf),
            static_cast<VkBuffer>(y_vkbuf),
            static_cast<VkBuffer>(poly_vkbuf),
            static_cast<VkBuffer>(out_vkbuf),
        },
        &pc, sizeof(pc), n
    );
}

int vksp_sjoin_polygons(
    void* x_vkbuf, size_t x_off,
    void* y_vkbuf, size_t y_off,
    int64_t n,
    void* verts_vkbuf,
    void* offsets_vkbuf,
    void* bboxes_vkbuf,
    int n_polys,
    void* out_vkbuf, size_t out_off
) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_st.ready || !x_vkbuf || !y_vkbuf || !verts_vkbuf || !offsets_vkbuf || !bboxes_vkbuf || !out_vkbuf ||
        n < 0 || n_polys <= 0 || (x_off % 4 != 0) || (y_off % 4 != 0) || (out_off % 4 != 0)) {
        return -1;
    }

    struct {
        uint32_t n;
        uint32_t n_polys;
        uint32_t x_base;
        uint32_t y_base;
        uint32_t out_base;
    } pc = {
        static_cast<uint32_t>(n),
        static_cast<uint32_t>(n_polys),
        static_cast<uint32_t>(x_off / 4),
        static_cast<uint32_t>(y_off / 4),
        static_cast<uint32_t>(out_off / 4),
    };

    return dispatch_pipe(
        g_st.p_sjoin,
        {
            static_cast<VkBuffer>(x_vkbuf),
            static_cast<VkBuffer>(y_vkbuf),
            static_cast<VkBuffer>(verts_vkbuf),
            static_cast<VkBuffer>(offsets_vkbuf),
            static_cast<VkBuffer>(bboxes_vkbuf),
            static_cast<VkBuffer>(out_vkbuf),
        },
        &pc, sizeof(pc), n
    );
}

int vksp_barnes_hut_step(
    void* src_pm_vkbuf,
    void* src_vg_vkbuf,
    void* dst_pm_vkbuf,
    void* dst_vg_vkbuf,
    void* leaf_heads_vkbuf,
    void* particle_next_vkbuf,
    void* tree_com_vkbuf,
    void* tree_size_vkbuf,
    uint32_t n,
    float G,
    float softening_sq,
    float dt,
    float damping,
    float theta,
    float restitution,
    float world_half
) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_st.ready || !src_pm_vkbuf || !src_vg_vkbuf || !dst_pm_vkbuf || !dst_vg_vkbuf ||
        !leaf_heads_vkbuf || !particle_next_vkbuf || !tree_com_vkbuf || !tree_size_vkbuf || n == 0u) {
        return -1;
    }

    VkBuffer spm = static_cast<VkBuffer>(src_pm_vkbuf);
    VkBuffer svg = static_cast<VkBuffer>(src_vg_vkbuf);
    VkBuffer dpm = static_cast<VkBuffer>(dst_pm_vkbuf);
    VkBuffer dvg = static_cast<VkBuffer>(dst_vg_vkbuf);
    VkBuffer lh  = static_cast<VkBuffer>(leaf_heads_vkbuf);
    VkBuffer pn  = static_cast<VkBuffer>(particle_next_vkbuf);
    VkBuffer tc  = static_cast<VkBuffer>(tree_com_vkbuf);
    VkBuffer ts  = static_cast<VkBuffer>(tree_size_vkbuf);

    if (g_st.sync_fn) g_st.sync_fn();

    update_set(g_st.p_bh_bin,    { spm, lh, pn });
    update_set(g_st.p_bh_leaves, { spm, lh, pn, tc, ts });
    update_set(g_st.p_bh_reduce, { tc, ts });
    update_set(g_st.p_bh_step,   { spm, svg, dpm, dvg, lh, pn, tc, ts });

    VkCommandBuffer cmd = g_st.cmd_buf;
    if (vkResetCommandBuffer(cmd, 0) != VK_SUCCESS) return -1;
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS) return -1;

    VkMemoryBarrier bar_xfer{};
    bar_xfer.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    bar_xfer.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bar_xfer.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

    VkMemoryBarrier bar_comp{};
    bar_comp.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    bar_comp.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bar_comp.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

    constexpr uint32_t NUM_LEAVES = 32768u;
    constexpr uint32_t LEAF_OFFSET = 4681u;

    // Pass 1: Clear leaf_heads to -1 (0xFFFFFFFF)
    vkCmdFillBuffer(cmd, lh, 0, NUM_LEAVES * sizeof(uint32_t), 0xFFFFFFFFu);
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &bar_xfer, 0, nullptr, 0, nullptr);

    // Pass 2: Bin particles into 32^3 Morton leaves
    struct { uint32_t n; float world_half; } pc_bin = { n, world_half };
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_st.p_bh_bin.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_st.p_bh_bin.pipe_layout, 0, 1, &g_st.p_bh_bin.desc_set, 0, nullptr);
    vkCmdPushConstants(cmd, g_st.p_bh_bin.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_bin), &pc_bin);
    vkCmdDispatch(cmd, (n + 63u) / 64u, 1, 1);
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &bar_comp, 0, nullptr, 0, nullptr);

    // Pass 3a: Build Level-5 leaves COM
    struct { uint32_t num_leaves; uint32_t leaf_offset; float leaf_size; } pc_leaf = { NUM_LEAVES, LEAF_OFFSET, (2.0f * world_half) / 32.0f };
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_st.p_bh_leaves.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_st.p_bh_leaves.pipe_layout, 0, 1, &g_st.p_bh_leaves.desc_set, 0, nullptr);
    vkCmdPushConstants(cmd, g_st.p_bh_leaves.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_leaf), &pc_leaf);
    vkCmdDispatch(cmd, (NUM_LEAVES + 63u) / 64u, 1, 1);
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &bar_comp, 0, nullptr, 0, nullptr);

    // Pass 3b: Bottom-up reduction L = 4, 3, 2, 1, 0
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_st.p_bh_reduce.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_st.p_bh_reduce.pipe_layout, 0, 1, &g_st.p_bh_reduce.desc_set, 0, nullptr);
    const uint32_t offsets[5] = { 585u, 73u, 9u, 1u, 0u };
    const uint32_t counts[5]  = { 4096u, 512u, 64u, 8u, 1u };
    for (int lv = 0; lv < 5; ++lv) {
        struct { uint32_t off; uint32_t cnt; } pc_red = { offsets[lv], counts[lv] };
        vkCmdPushConstants(cmd, g_st.p_bh_reduce.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_red), &pc_red);
        vkCmdDispatch(cmd, (counts[lv] + 63u) / 64u, 1, 1);
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &bar_comp, 0, nullptr, 0, nullptr);
    }

    // Pass 4: Barnes-Hut traversal + elastic collisions + integration
    struct {
        uint32_t n;
        uint32_t leaf_offset;
        float G;
        float softening_sq;
        float dt;
        float damping;
        float theta_sq;
        float restitution;
        float world_half;
    } pc_step = { n, LEAF_OFFSET, G, softening_sq, dt, damping, theta * theta, restitution, world_half };
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_st.p_bh_step.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_st.p_bh_step.pipe_layout, 0, 1, &g_st.p_bh_step.desc_set, 0, nullptr);
    vkCmdPushConstants(cmd, g_st.p_bh_step.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_step), &pc_step);
    vkCmdDispatch(cmd, (n + 63u) / 64u, 1, 1);

    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) return -1;
    if (g_st.submit_fn(cmd, g_st.fence) != 0) return -1;
    VkResult wrc = vkWaitForFences(g_st.dev, 1, &g_st.fence, VK_TRUE, UINT64_MAX);
    VkResult rrc = vkResetFences(g_st.dev, 1, &g_st.fence);
    return (wrc == VK_SUCCESS && rrc == VK_SUCCESS) ? 0 : -1;
}

} // extern "C"
