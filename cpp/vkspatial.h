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

#ifdef __cplusplus
}
#endif

#endif /* VKSPATIAL_H */
