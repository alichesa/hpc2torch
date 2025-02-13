#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <cstdint>

constexpr int THREADS_PER_BLOCK = 256;

// 用宏生成 axis==0 专用 kernel（每个 block 处理一行）
// type 为数据类型，suffix 用于函数名后缀
#define DEFINE_GATHER_AXIS0_KERNEL(type, suffix)                                \
__global__ void gather_axis0_kernel_##suffix(const type* __restrict__ input,     \
                                             const int64_t* __restrict__ indices,  \
                                             type* __restrict__ output,            \
                                             long mid, long inner) {               \
    int idx = blockIdx.x;                                                       \
    int tid = threadIdx.x;                                                      \
    int64_t index_val = indices[idx];                                           \
    if (index_val < 0 || index_val >= mid) return;                              \
    long input_offset  = index_val * inner;                                     \
    long output_offset = idx * inner;                                           \
    for (int j = tid; j < inner; j += blockDim.x) {                             \
        output[output_offset + j] = input[input_offset + j];                    \
    }                                                                           \
}

// 用宏生成统一 kernel（适用于 axis!=0 或 outer!=1 的情况）
// 输出视作 (outer, index_num, inner) 三维结构
#define DEFINE_GATHER_KERNEL(type, suffix)                                      \
__global__ void gather_kernel_##suffix(const type* __restrict__ input,          \
                                       const int64_t* __restrict__ indices,       \
                                       type* __restrict__ output,                 \
                                       long outer, long mid, long inner,          \
                                       long index_num) {                          \
    long total = outer * index_num * inner;                                     \
    long idx = blockIdx.x * blockDim.x + threadIdx.x;                           \
    if (idx >= total) return;                                                   \
    long inner_dim = index_num * inner;                                         \
    long o   = idx / inner_dim;                                                 \
    long rem = idx % inner_dim;                                                 \
    long i_idx = rem / inner;                                                   \
    long k   = rem % inner;                                                     \
    int64_t index_val = indices[i_idx];                                         \
    if (index_val < 0 || index_val >= mid) return;                              \
    long input_idx = o * (mid * inner) + index_val * inner + k;                 \
    output[idx] = input[input_idx];                                             \
}

///////////////////////////
// 生成 f32 版本的 kernel
DEFINE_GATHER_AXIS0_KERNEL(float, f32)
DEFINE_GATHER_KERNEL(float, f32)

///////////////////////////
// 生成 f16 版本的 kernel
DEFINE_GATHER_AXIS0_KERNEL(half, f16)
DEFINE_GATHER_KERNEL(half, f16)

///////////////////////////
// 外部接口函数
// 参数说明：
//  - outer: 输入张量在 axis 之前所有维度的乘积
//  - mid  : 输入张量在 axis 上的尺寸
//  - inner: 输入张量在 axis 之后所有维度的乘积
//  - index_num: indices 张量展平后的元素数
//  - zijie: 元素字节数（4 表示 float32，2 表示 float16）
extern "C" void my_gather(void const *input, void const *indices, void *output,
                            long outer, long mid, long inner,
                            long index_num, long zijie) {
    cudaError_t err;
    // 当 outer==1 时（即 axis==0 且 outer==1）使用专用 kernel
    if (outer == 1) {
        int blocks = index_num;  // 每个 block 负责一行
        if (zijie == 4) {
            gather_axis0_kernel_f32<<<blocks, THREADS_PER_BLOCK>>>(
                reinterpret_cast<const float*>(input),
                reinterpret_cast<const int64_t*>(indices),
                reinterpret_cast<float*>(output),
                mid, inner
            );
        } else if (zijie == 2) {
            gather_axis0_kernel_f16<<<blocks, THREADS_PER_BLOCK>>>(
                reinterpret_cast<const half*>(input),
                reinterpret_cast<const int64_t*>(indices),
                reinterpret_cast<half*>(output),
                mid, inner
            );
        } else {
            printf("Error with type\n");
            return;
        }
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
        }
        return;
    }

    // 对于其他情况，使用统一 kernel
    long total = outer * index_num * inner;
    int blocks = (total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (zijie == 4) {
        gather_kernel_f32<<<blocks, THREADS_PER_BLOCK>>>(
            reinterpret_cast<const float*>(input),
            reinterpret_cast<const int64_t*>(indices),
            reinterpret_cast<float*>(output),
            outer, mid, inner, index_num
        );
    } else if (zijie == 2) {
        gather_kernel_f16<<<blocks, THREADS_PER_BLOCK>>>(
            reinterpret_cast<const half*>(input),
            reinterpret_cast<const int64_t*>(indices),
            reinterpret_cast<half*>(output),
            outer, mid, inner, index_num
        );
    } else {
        printf("Error with type\n");
        return;
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
}
