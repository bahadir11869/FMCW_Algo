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

__global__ void transpose_multi_channel_kernel(cuComplex* idata, cuComplex* odata, int width, int height) {
    // 1. Kanal Ofsetini Hesapla (MIMO için en önemli kısım)
    // Her kanalın verisi bellekte ardışık: ChannelSize = width * height
    int ch = blockIdx.z; 
    int channel_offset = ch * width * height;

    // İşlem yapılacak kanalın başlangıç adreslerini belirle
    cuComplex* current_idata = idata + channel_offset;
    cuComplex* current_odata = odata + channel_offset;

    // 2. Shared Memory Tanımla (Bank conflict önlemek için +1 padding)
    __shared__ cuComplex tile[TILE_DIM][TILE_DIM + 1];

    // 3. Global Memory -> Shared Memory (Okuma)
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += NUM_CHANNELS) {
        if (x < width && (y + j) < height) {
            tile[threadIdx.y + j][threadIdx.x] = current_idata[(y + j) * width + x];
        }
    }

    __syncthreads();

    // 4. Shared Memory -> Global Memory (Yazma - Transpoze edilmiş halde)
    // Yeni x ve y koordinatlarını transpoze ederek hesapla
    x = blockIdx.y * TILE_DIM + threadIdx.x;  // Orijinal y blok indisi
    y = blockIdx.x * TILE_DIM + threadIdx.y;  // Orijinal x blok indisi

    for (int j = 0; j < TILE_DIM; j += NUM_CHANNELS) {
        if (x < height && (y + j) < width) {
            current_odata[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

__global__ void sumChannelsKernel(cuComplex* d_all_channels, cuComplex* d_output_sum, int num_channels, int channel_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < channel_size) {
        cuComplex sum = make_cuComplex(0.0f, 0.0f);
        for (int ch = 0; ch < num_channels; ++ch) {
            cuComplex val = d_all_channels[ch * channel_size + idx];
            sum.x += val.x; // Real
            sum.y += val.y; // Imag
        }
        d_output_sum[idx] = sum;
    }
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
        float angle = -2.0 * PI * pair_idx / (float)(current_stage_width * 2.0);
        cuComplex w = make_cuComplex(cosf(angle), sinf(angle));

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


__global__ void k_fft_shared(cuComplex* d_data, int n, int log2_n) 
{
    extern __shared__ cuComplex s_data[];

    int tid = threadIdx.x;
    int batch_idx = blockIdx.x; 

    int offset = batch_idx * n;

    // --- 1. ADIM: YÜKLEME VE BIT-REVERSAL ---
    int idx1 = tid;
    int idx2 = tid + (n >> 1); // n/2

    unsigned int rev_idx1 = gpu_bit_reverse(idx1, log2_n);
    unsigned int rev_idx2 = gpu_bit_reverse(idx2, log2_n);

    s_data[rev_idx1] = d_data[offset + idx1];
    s_data[rev_idx2] = d_data[offset + idx2];

    __syncthreads();

    // --- 2. ADIM: BUTTERFLY AŞAMALARI ---
    for (int s = 1; s <= log2_n; ++s) 
    {
        int m = 1 << s;        
        int m2 = m >> 1;       
        
        // HIZLI INDEKSLEME (Bitwise işlemler)
        int k = tid & (m2 - 1);         // tid % m2 yerine
        int j = ((tid - k) << 1) + k;   // (tid / m2) * m + k yerine
        
        // --- KRİTİK PERFORMANS DÜZELTMESİ ---
        // Double yerine Float kullanıyoruz:
        // PI sabiti defines.h içinde float tanımlıydı, burada da float literaller (2.0f) kullanıyoruz.
        float angle = -2.0f * PI * (float)k / (float)m;
        
        float c, s_val;
        // GPU'nun özel trigonometri ünitesini (SFU) kullanan hızlı fonksiyon:
        __sincosf(angle, &s_val, &c); 
        
        cuComplex w = make_cuComplex(c, s_val);
        // ------------------------------------

        cuComplex data1 = s_data[j];
        cuComplex data2 = s_data[j + m2];

        cuComplex w_d2 = cuCmulf(w, data2);
        
        s_data[j]      = cuCaddf(data1, w_d2);
        s_data[j + m2] = cuCsubf(data1, w_d2);
        
        __syncthreads(); 
    }

    // --- 3. ADIM: YAZMA ---
    d_data[offset + idx1] = s_data[idx1];
    d_data[offset + idx2] = s_data[idx2];
}