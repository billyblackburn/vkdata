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

/**
 * Zero-copy GPU line/trajectory rasterizer (NaN-separated polylines).
 * mode: 0=count, 1=mean (vals_vkbuf=float32), 2=cat (vals_vkbuf=int32, palette_vkbuf=256x uint32 RGBA).
 */
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
);

/**
 * Zero-copy GPU shading for mean (mode=1 fire, mode=2 fire_r) or categorical RGB (mode=3)
 * grids produced by vks_raster_lines(), with optional bilinear GPU basemap compositing.
 */
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
);

/**
 * Zero-copy GPU 3D perspective point/line rasterizer.
 * mode: 0=points(fire by z), 1=lines(fire by z), 2=lines(cat RGB), 3=points(cat RGB).
 * cam_vkbuf holds 24 float32 camera + bounds parameters.
 */
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
);

/**
 * Zero-copy GPU 3D volumetric compositor + perspective ground-plane Esri satellite map.
 */
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
);

#ifdef __cplusplus
}
#endif

#endif /* VKSHADER_H */
