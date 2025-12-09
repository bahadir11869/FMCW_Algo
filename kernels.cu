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


__device__ unsigned int gpu_bit_reverse(unsigned int k, int num_bits) {
    unsigned int r = 0;
    for (int i = 0; i < num_bits; i++) {
        r = (r << 1) | (k & 1);
        k >>= 1;
    }
    return r;
}

__global__ void k_bit_reversal(cuComplex* d_in, cuComplex* d_out, int n, int num_bits, int batch_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y;

    if (idx < n && batch_idx < batch_count) {
        unsigned int reversed_idx = gpu_bit_reverse(idx, num_bits);
        
        // Batch ofseti (Hangi satırdayız?)
        int in_pos = batch_idx * n + idx;
        int out_pos = batch_idx * n + reversed_idx;
        
        d_out[out_pos] = d_in[in_pos];
    }
}

__global__ void k_butterfly_stage(cuComplex* d_data, int n, int current_stage_width, int batch_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y;

    int half_n = n >> 1; // n/2 tane kelebek var

    if (idx < half_n && batch_idx < batch_count) {
        // Hangi gruptayız? (Recursive ağacın hangi dalı?)
        int group_idx = idx / current_stage_width;
        int pair_idx  = idx % current_stage_width;

        int pos1 = batch_idx * n + (group_idx * 2 * current_stage_width + pair_idx);
        int pos2 = pos1 + current_stage_width;

        // Twiddle Factor Hesabı: W = e^(-j * 2*pi * k / N_stage)
        double angle = -2.0 * PI * pair_idx / (double)(current_stage_width * 2.0);
        cuComplex w = make_cuComplex(cos(angle), sin(angle));

        cuComplex data1 = d_data[pos1];
        cuComplex data2 = d_data[pos2];

        // Kelebek Matematigi
        cuComplex w_d2 = cuCmulf(w, data2);
        
        // A = A + W*B
        d_data[pos1] = cuCaddf(data1, w_d2);
        // B = A - W*B
        d_data[pos2] = cuCsubf(data1, w_d2);
    }
}
