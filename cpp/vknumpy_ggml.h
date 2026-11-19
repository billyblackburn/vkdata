#ifndef VKNUMPY_GGML_H
#define VKNUMPY_GGML_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    VKN_DTYPE_F32 = 0,
    VKN_DTYPE_F16 = 1,
    VKN_DTYPE_I32 = 2,
} VknDtype;

typedef enum {
    VKN_BIN_ADD = 0,
    VKN_BIN_SUB = 1,
    VKN_BIN_MUL = 2,
    VKN_BIN_DIV = 3,
} VknBinaryOp;

typedef enum {
    VKN_UNARY_NEG     = 0,
    VKN_UNARY_ABS     = 1,
    VKN_UNARY_SQR     = 2,
    VKN_UNARY_SQRT    = 3,
    VKN_UNARY_EXP     = 4,
    VKN_UNARY_LOG     = 5,
    VKN_UNARY_SIN     = 6,
    VKN_UNARY_COS     = 7,
    VKN_UNARY_TANH    = 8,
    VKN_UNARY_RELU    = 9,
    VKN_UNARY_SIGMOID = 10,
    VKN_UNARY_SILU    = 11,
    VKN_UNARY_GELU    = 12,
} VknUnaryOp;

typedef struct VknBuffer VknBuffer;

/* Lifecycle & device info */
int         vkn_init(void);
void        vkn_shutdown(void);
int         vkn_is_vulkan(void);
const char* vkn_device_name(void);
size_t      vkn_device_free_memory(void);
size_t      vkn_device_total_memory(void);
const char* vkn_last_error(void);

/* Buffer management */
VknBuffer*  vkn_buf_alloc(size_t nbytes);
void        vkn_buf_free(VknBuffer* buf);
size_t      vkn_buf_size(const VknBuffer* buf);
int         vkn_buf_upload(VknBuffer* buf, size_t offset, const void* src, size_t nbytes);
int         vkn_buf_download(const VknBuffer* buf, size_t offset, void* dst, size_t nbytes);
int         vkn_buf_copy(const VknBuffer* src, size_t src_off, VknBuffer* dst, size_t dst_off, size_t nbytes);

/* GEMM & BLAS */
int vkn_matmul(
    const VknBuffer* a, size_t a_off, const int64_t* a_shape, int a_ndim,
    const VknBuffer* b, size_t b_off, const int64_t* b_shape, int b_ndim,
    int b_transposed,
    VknBuffer* out, size_t out_off,
    int dtype
);

int vkn_dot_1d(
    const VknBuffer* a, size_t a_off,
    const VknBuffer* b, size_t b_off,
    int64_t n,
    VknBuffer* out, size_t out_off,
    int dtype
);

int vkn_outer(
    const VknBuffer* a, size_t a_off, int64_t m,
    const VknBuffer* b, size_t b_off, int64_t n,
    VknBuffer* out, size_t out_off,
    int dtype
);

/* Elementwise ops */
int vkn_binary_op(
    int op,
    const VknBuffer* a, size_t a_off, const int64_t* a_shape,
    const VknBuffer* b, size_t b_off, const int64_t* b_shape,
    VknBuffer* out, size_t out_off, const int64_t* out_shape,
    int ndim,
    int dtype
);

int vkn_unary_op(
    int op,
    const VknBuffer* a, size_t a_off,
    int64_t n,
    VknBuffer* out, size_t out_off,
    int dtype
);

int vkn_scale_bias(
    const VknBuffer* a, size_t a_off,
    int64_t n,
    float scale,
    float bias,
    VknBuffer* out, size_t out_off,
    int dtype
);

int vkn_clamp(
    const VknBuffer* a, size_t a_off,
    int64_t n,
    float min_val,
    float max_val,
    VknBuffer* out, size_t out_off,
    int dtype
);

/* Reductions */
int vkn_sum_all(
    const VknBuffer* a, size_t a_off,
    int64_t n,
    VknBuffer* out, size_t out_off,
    int dtype
);

int vkn_sum_rows(
    const VknBuffer* a, size_t a_off,
    int64_t rows, int64_t cols,
    VknBuffer* out, size_t out_off,
    int dtype
);

int vkn_mean_rows(
    const VknBuffer* a, size_t a_off,
    int64_t rows, int64_t cols,
    VknBuffer* out, size_t out_off,
    int dtype
);

int vkn_argmax_rows(
    const VknBuffer* a, size_t a_off,
    int64_t rows, int64_t cols,
    VknBuffer* out_i32, size_t out_off,
    int dtype
);

/* Layout & Cast */
int vkn_transpose_cont(
    const VknBuffer* a, size_t a_off,
    const int64_t* shape, const int* perm, int ndim,
    VknBuffer* out, size_t out_off,
    int dtype
);

int vkn_cast(
    const VknBuffer* a, size_t a_off, int src_dtype,
    int64_t n,
    VknBuffer* out, size_t out_off, int dst_dtype
);

#ifdef __cplusplus
}
#endif

#endif /* VKNUMPY_GGML_H */
