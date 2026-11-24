#ifndef VKSPATIAL_H
#define VKSPATIAL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int  (*VkspSubmitFn)(void* cmd_buf, void* fence);
typedef void (*VkspSyncFn)(void);

int vksp_init(
    void* vk_device,
    uint32_t compute_queue_family,
    VkspSubmitFn submit_fn,
    VkspSyncFn sync_fn
);

int vksp_morton_2d(
    void* x_vkbuf, size_t x_off,
    void* y_vkbuf, size_t y_off,
    int64_t n,
    float xmin, float xmax,
    float ymin, float ymax,
    void* out_vkbuf, size_t out_off
);

int vksp_point_in_polygon(
    void* x_vkbuf, size_t x_off,
    void* y_vkbuf, size_t y_off,
    int64_t n,
    void* poly_vkbuf, size_t poly_off,
    int n_verts,
    void* out_vkbuf, size_t out_off
);

int vksp_sjoin_polygons(
    void* x_vkbuf, size_t x_off,
    void* y_vkbuf, size_t y_off,
    int64_t n,
    void* verts_vkbuf,
    void* offsets_vkbuf,
    void* bboxes_vkbuf,
    int n_polys,
    void* out_vkbuf, size_t out_off
);

/**
 * Single-command-buffer 6-level 3D Barnes-Hut Morton octree N-body step + elastic collisions.
 */
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
);

/**
 * Compile (cached by SPIR-V hash) and dispatch an arbitrary user SPIR-V compute shader
 * zero-copy on raw VkBuffers with optional push constants and 3D workgroup grid.
 */
int vksp_dispatch_spv(
    const uint32_t* spv_words,
    size_t spv_bytes,
    void** vkbufs,
    uint32_t num_bufs,
    const void* push_data,
    uint32_t push_bytes,
    uint32_t gx,
    uint32_t gy,
    uint32_t gz
);

#ifdef __cplusplus
}
#endif

#endif /* VKSPATIAL_H */
