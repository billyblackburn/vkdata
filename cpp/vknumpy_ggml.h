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
    VKN_UNARY_FLOOR   = 13,
    VKN_UNARY_CEIL    = 14,
    VKN_UNARY_ROUND   = 15,
    VKN_UNARY_TRUNC   = 16,
    VKN_UNARY_SGN     = 17,
    VKN_UNARY_EXPM1   = 18,
    VKN_UNARY_STEP    = 19,
} VknUnaryOp;

typedef enum {
    VKN_TRI_UPPER_DIAG = 0,
    VKN_TRI_UPPER      = 1,
    VKN_TRI_LOWER_DIAG = 2,
    VKN_TRI_LOWER      = 3,
} VknTriType;

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

/* Phase 1: Native ggml-vulkan creation, manipulation, sorting, gather & triangular solve */
int vkn_fill(
    VknBuffer* out, size_t out_off,
    int64_t n,
    float value,
    int dtype
);

int vkn_arange(
    float start, float stop, float step,
    VknBuffer* out, size_t out_off,
    int dtype
);

int vkn_diag(
    const VknBuffer* a, size_t a_off,
    int64_t n,
    VknBuffer* out, size_t out_off,
    int dtype
);

int vkn_tri(
    const VknBuffer* a, size_t a_off,
    const int64_t* shape, int ndim,
    int tri_type,
    VknBuffer* out, size_t out_off,
    int dtype
);

int vkn_concat(
    const VknBuffer* a, size_t a_off, const int64_t* a_shape,
    const VknBuffer* b, size_t b_off, const int64_t* b_shape,
    int ndim, int axis,
    VknBuffer* out, size_t out_off,
    int dtype
);

int vkn_pad(
    const VknBuffer* a, size_t a_off,
    const int64_t* shape, int ndim,
    const int* pad_before, const int* pad_after,
    VknBuffer* out, size_t out_off,
    int dtype
);

int vkn_roll(
    const VknBuffer* a, size_t a_off,
    const int64_t* shape, int ndim,
    const int* shifts,
    VknBuffer* out, size_t out_off,
    int dtype
);

int vkn_repeat(
    const VknBuffer* a, size_t a_off, const int64_t* a_shape,
    VknBuffer* out, size_t out_off, const int64_t* out_shape,
    int ndim,
    int dtype
);

int vkn_cumsum_rows(
    const VknBuffer* a, size_t a_off,
    int64_t rows, int64_t cols,
    VknBuffer* out, size_t out_off,
    int dtype
);

int vkn_argsort_rows(
    const VknBuffer* a, size_t a_off,
    int64_t rows, int64_t cols,
    int descending,
    VknBuffer* out_i32, size_t out_off,
    int dtype
);

int vkn_top_k_rows(
    const VknBuffer* a, size_t a_off,
    int64_t rows, int64_t cols,
    int k,
    VknBuffer* out_i32, size_t out_off,
    int dtype
);

int vkn_get_rows(
    const VknBuffer* a, size_t a_off,
    int64_t rows, int64_t cols,
    const VknBuffer* idx_i32, size_t idx_off,
    int64_t n_idx,
    VknBuffer* out, size_t out_off,
    int dtype
);

int vkn_solve_tri(
    const VknBuffer* a, size_t a_off, const int64_t* a_shape,
    const VknBuffer* b, size_t b_off, const int64_t* b_shape,
    int ndim,
    int left, int lower, int unit_diag,
    VknBuffer* out, size_t out_off,
    int dtype
);

#ifdef __cplusplus
}
#endif

#endif /* VKNUMPY_GGML_H */
