// #include <cuda.h>
// #include <cub/cub.cuh>
// #include <cuda_fp16.h>  

// constexpr long BLOCKdim = 128;


// __global__
// void gatherkernel_f32(float const *input, int64_t const *indices,
//                       float *output, long Step_size, long outputsize)
// {
//     long long elementsPerThread = (Step_size + BLOCKdim - 1) / BLOCKdim;
//     long long outputBase = (blockIdx.x * gridDim.y + blockIdx.y) * Step_size;
//     long long inputBase = (blockIdx.x * gridDim.y + indices[blockIdx.y]) * Step_size;

//     for (int i = 0; i < elementsPerThread; i++)
//     {
//         long long localIndex = threadIdx.x + i * BLOCKdim;
//         if (localIndex >= Step_size || outputBase + localIndex >= outputsize)
//             break;

//         output[outputBase + localIndex] = input[inputBase + localIndex];
//     }
// }


// __global__ 
// void gatherkernel_f16(half const *input, int64_t const *indices,
//                       half *output, long Step_size, long outputsize)
// {
//     long long elementsPerThread = (Step_size + BLOCKdim - 1) / BLOCKdim;
//     long long outputBase = (blockIdx.x * gridDim.y + blockIdx.y) * Step_size;
//     long long inputBase = (blockIdx.x * gridDim.y + indices[blockIdx.y]) * Step_size;

//     for (int i = 0; i < elementsPerThread; i++) {
//         long long localIndex = threadIdx.x + i * BLOCKdim;

        
//         if (localIndex * 2 + 1 < Step_size && outputBase + localIndex * 2 + 1 < outputsize) {
//             half2 value = *reinterpret_cast<const half2*>(&input[inputBase + localIndex * 2]);
//             *reinterpret_cast<half2*>(&output[outputBase + localIndex * 2]) = value;
//         }
        
//         else if (localIndex * 2 < Step_size && outputBase + localIndex * 2 < outputsize) {
//             output[outputBase + localIndex * 2] = input[inputBase + localIndex * 2];
//         }
//     }
// }


// extern "C" void my_gather(void const *input, void const *indices, void *output, long Step_size, long axis_num, long inputSize, long indexSize, long zijie, long outputsize)
// {
//     long griddim_x = inputSize / axis_num / Step_size;
//     long griddim_y = indexSize;

//     dim3 GRIDdim(griddim_x, griddim_y);

//     if (zijie == 4) {
//         gatherkernel_f32<<<GRIDdim, BLOCKdim>>>(
//             reinterpret_cast<float const *>(input),
//             reinterpret_cast<int64_t const *>(indices),
//             reinterpret_cast<float *>(output),
//             Step_size,
//             outputsize
//         );
//     } 
//     else if (zijie == 2) {
//         gatherkernel_f16<<<GRIDdim, BLOCKdim>>>(
//             reinterpret_cast<half const *>(input),
//             reinterpret_cast<int64_t const *>(indices),
//             reinterpret_cast<half *>(output),
//             Step_size,
//             outputsize
//         );
//     } 
//     else {
//         printf("Error with type");
//         return;
//     }

//     // 检查 CUDA 错误
//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         printf("CUDA error: %s\n", cudaGetErrorString(err));
//     }
// }