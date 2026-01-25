// Dosya Adı: Kernels.h
#pragma once
#include "../defines.h"

// Kernel Prototipi (Gövdesi yok, sadece imzası)
__global__ void transpose_optimized_kernel(cuComplex* input, cuComplex* output, int width, int height);
__global__ void transpose_multi_channel_kernel(cuComplex* idata, cuComplex* odata, int width, int height); 
__global__ void sumChannelsKernel(cuComplex* d_all_channels, cuComplex* d_output_sum, int num_channels, int channel_size);
__device__ unsigned int gpu_bit_reverse(unsigned int k, int num_bits);
__global__ void k_bit_reversal(cuComplex* d_in, cuComplex* d_out, int n, int num_bits, int batch_count);
__global__ void k_fft_shared(cuComplex* d_data, int n, int log2_n);
__global__ void k_butterfly_stage(cuComplex* d_data, int n, int current_stage_width, int batch_count); 
