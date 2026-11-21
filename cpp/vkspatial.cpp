#include "vkspatial.h"

#include <vulkan/vulkan.h>

#include "vksp_spv_morton_2d.h"
#include "vksp_spv_point_in_polygon.h"
#include "vksp_spv_sjoin_polygons.h"

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
    pool_size.descriptorCount = 32;

    VkDescriptorPoolCreateInfo dp_ci{};
    dp_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dp_ci.maxSets = 8;
    dp_ci.poolSizeCount = 1;
    dp_ci.pPoolSizes = &pool_size;
    if (vkCreateDescriptorPool(g_st.dev, &dp_ci, nullptr, &g_st.desc_pool) != VK_SUCCESS) return -1;

    bool ok1 = create_pipe(g_st, vksp_spv_morton_2d, sizeof(vksp_spv_morton_2d), 3, 8 * sizeof(uint32_t), g_st.p_morton);
    bool ok2 = create_pipe(g_st, vksp_spv_point_in_polygon, sizeof(vksp_spv_point_in_polygon), 4, 6 * sizeof(uint32_t), g_st.p_pip);
    bool ok3 = create_pipe(g_st, vksp_spv_sjoin_polygons, sizeof(vksp_spv_sjoin_polygons), 6, 5 * sizeof(uint32_t), g_st.p_sjoin);

    g_st.ready = ok1 && ok2 && ok3;
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

} // extern "C"
