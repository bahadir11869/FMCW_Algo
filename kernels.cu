// Dosya Adı: Kernels.cu
#include "kernels.h"

// Kernel Gövdesi
__global__ void transpose_optimized_kernel(cuComplex* input, cuComplex* output, int width, int height) 
{
    __shared__ cuComplex tile[TILE_DIM][TILE_DIM + 1];
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    if (x < width && y < height) tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    __syncthreads();
    x = blockIdx.y * TILE_DIM + threadIdx.x; 
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    if (x < height && y < width) output[y * height + x] = tile[threadIdx.x][threadIdx.y];
}
