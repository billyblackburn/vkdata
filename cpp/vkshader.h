#ifndef VKSHADER_H
#define VKSHADER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int  (*VksSubmitFn)(void* cmd_buf, void* fence);
typedef void (*VksSyncFn)(void);

/**
 * Initialize vkshader on an existing VkDevice / compute queue (e.g. from vknumpy).
 * Must be called once before vks_shade_rgba().
 */
int vks_init(
    void* vk_device,
    uint32_t compute_queue_family,
    VksSubmitFn submit_fn,
    VksSyncFn sync_fn
);

/**
 * Shade a 2D int32 count grid with eq_hist/log/cbrt/linear + Fire colormap +
 * optional bilinear GPU basemap compositing, directly on shared VkBuffers.
 *
 * stats_vkbuf must hold at least 2050 * 4 bytes; cdf_vkbuf at least 2048 * 4 bytes.
 */
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
);

#ifdef __cplusplus
}
#endif

#endif /* VKSHADER_H */
