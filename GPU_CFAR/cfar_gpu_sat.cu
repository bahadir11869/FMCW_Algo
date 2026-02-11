// cfar_gpu_sat.cu (FINAL OPTIMIZED VERSION)
#include "cfar_gpu_sat.hpp"
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cmath>

#define CUDA_CHECK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while (0)

__device__ __forceinline__ size_t didx(int r, int c, int C) {
    return (size_t)r * C + c;
}

// LDG Wrapper: Salt okunur veriyi Texture Cache'den �eker
// (Detect a�amas� ve Row scan okumas� i�in kritik)
__device__ __forceinline__ float load_ro(const float* __restrict__ ptr) {
#if __CUDA_ARCH__ >= 350
    return __ldg(ptr);
#else
    return *ptr;
#endif
}

// =======================================================
// 1. SATIR KERNEL (WARP SCAN OPTIMIZATION)
// =======================================================
// Not: Sat�r verisi bellekte ayr�k durdu�u i�in (strided),
// tek thread gezmek yerine Warp i�indeki threadler yan yana okuyup
// veriyi kendi aralar�nda (Shuffle) takas ederler.

__device__ __forceinline__ float warp_scan_add(float val) {
    // 32 thread i�inde k�m�latif toplam (Inclusive Scan)
    unsigned int mask = 0xffffffff;
    float tmp = __shfl_up_sync(mask, val, 1);
    if (threadIdx.x % 32 >= 1) val += tmp;

    tmp = __shfl_up_sync(mask, val, 2);
    if (threadIdx.x % 32 >= 2) val += tmp;

    tmp = __shfl_up_sync(mask, val, 4);
    if (threadIdx.x % 32 >= 4) val += tmp;

    tmp = __shfl_up_sync(mask, val, 8);
    if (threadIdx.x % 32 >= 8) val += tmp;

    tmp = __shfl_up_sync(mask, val, 16);
    if (threadIdx.x % 32 >= 16) val += tmp;

    return val;
}

__global__ void sat_row_prefix_warp(
    const float* __restrict__ in,
    float* __restrict__ sat,
    int rows, int cols)
{
    // Bir WARP (32 thread), tek bir SATIRI i�ler.
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32; // Warp i�indeki s�ra

    // Hangi sat�r� i�leyece�iz?
    int r = blockIdx.x * (blockDim.x / 32) + warp_id;

    if (r >= rows) return;

    size_t row_offset = (size_t)r * cols;
    float running_sum = 0.0f;

    // Sat�r� 32'lik paketler halinde gez
    for (int c = 0; c < cols; c += 32) {
        int col_idx = c + lane_id;
        float val = 0.0f;

        // 1. Coalesced Load (H�zl� Okuma)
        if (col_idx < cols) {
            val = load_ro(&in[row_offset + col_idx]);
        }

        // 2. Warp i�i Scan (Registerlarda)
        val = warp_scan_add(val);

        // 3. �nceki paketten devir
        val += running_sum;

        // 4. Coalesced Store (H�zl� Yazma)
        if (col_idx < cols) {
            sat[row_offset + col_idx] = val;
        }

        // 5. Son thread (31) elindeki toplam� sonrakine devreder
        float last_val = __shfl_sync(0xffffffff, val, 31);
        running_sum = last_val;
    }
}

// =======================================================
// 2. S�TUN KERNEL (CLASSIC - ZATEN OPT�MAL)
// =======================================================
// Not: Threadler yan yana s�tunlar� okudu�u i�in eri�im zaten
// "Coalesced" (biti�ik) durumdad�r. Warp shuffle gerekmez.

__global__ void sat_col_prefix(
    float* __restrict__ sat,
    int rows, int cols)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cols) return;

    float sum = 0.0f;
    for (int r = 0; r < rows; ++r) {
        size_t id = didx(r, c, cols);
        // Read-Modify-Write oldu�u i�in burada ldg kullan�lmaz
        sum += sat[id];
        sat[id] = sum;
    }
}

// =======================================================
// 3. DETECT KERNEL (TEXTURE CACHE OPTIMIZED)
// =======================================================

__device__ __forceinline__
float sat_rect_sum(const float* __restrict__ sat,
    int rows, int cols,
    int r0, int c0, int r1, int c1)
{
    // Texture cache (Read-Only) kullan�m� - En �nemli h�zland�rma buras�
    float A = load_ro(&sat[didx(r1, c1, cols)]);
    float B = (c0 > 0) ? load_ro(&sat[didx(r1, c0 - 1, cols)]) : 0.0f;
    float C = (r0 > 0) ? load_ro(&sat[didx(r0 - 1, c1, cols)]) : 0.0f;
    float D = (r0 > 0 && c0 > 0) ? load_ro(&sat[didx(r0 - 1, c0 - 1, cols)]) : 0.0f;
    return A - B - C + D;
}

__global__ void cfar_detect_sat(
    const float* __restrict__ sat,
    const float* __restrict__ power,
    bool* __restrict__ detect,
    int rows, int cols,
    int win_r, int win_c,
    int guard_r, int guard_c,
    int Nref, float alpha)
{
    // Shared memory atomiclerini h�zland�rmak i�in
    __shared__ unsigned int s_det, s_tp, s_fp;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_det = 0; s_tp = 0; s_fp = 0;
    }
    __syncthreads();

    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check
    if (r >= win_r && r < (rows - win_r) &&
        c >= win_c && c < (cols - win_c))
    {
        // 1. SAT �zerinden HIZLI pencere toplam� (__ldg ile)
        float sum_big = sat_rect_sum(sat, rows, cols, r - win_r, c - win_c, r + win_r, c + win_c);
        float sum_g = sat_rect_sum(sat, rows, cols, r - guard_r, c - guard_c, r + guard_r, c + guard_c);

        float ref = sum_big - sum_g;
        float thr = alpha * (ref / (float)Nref);

        // Power haritas�n� da texture cache'den okuyal�m
        size_t idx_curr = didx(r, c, cols);
        float cur = load_ro(&power[idx_curr]);

        bool out = (cur > thr) ? true : false;

        if (detect) detect[idx_curr] = out;


    }
}

// =======================================================
// HOST IMPLEMENTATION
// =======================================================

void GPUCFAR_SAT::ensure_device_alloc(size_t Npix) {

    CUDA_CHECK(cudaMalloc(&d_sat, Npix * sizeof(float)));

}

void GPUCFAR_SAT::init(const CFARParams& p) {
    rows = p.rows; cols = p.cols;
    ref_r = p.ref_r; ref_c = p.ref_c;
    guard_r = p.guard_r; guard_c = p.guard_c;
    win_r = ref_r + guard_r;
    win_c = ref_c + guard_c;

    const int big_h = 2 * win_r + 1;
    const int big_w = 2 * win_c + 1;
    const int guard_h = 2 * guard_r + 1;
    const int guard_w = 2 * guard_c + 1;
    Nref = big_h * big_w - guard_h * guard_w;
    alpha = Nref * (std::pow(p.pfa, -1.0f / Nref) - 1.0f);

    CUDA_CHECK(cudaFree(0)); // Cihaz� uyand�r

    const size_t Npix = (size_t)rows * cols;
    ensure_device_alloc(Npix);

    // Cache Config (L1 vs Shared Memory)
    // Row prefix i�in art�k shared memory kullanm�yoruz (shuffle kullan�yoruz), L1 tercih edebiliriz.
    cudaFuncSetCacheConfig(sat_row_prefix_warp, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(sat_col_prefix, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(cfar_detect_sat, cudaFuncCachePreferL1);
}


void GPUCFAR_SAT::process(const CFARParams& p, float* d, bool* b)
{
    int threads = 256;
    int warps_per_block = threads / 32;

    dim3 grid_rows((rows + warps_per_block - 1) / warps_per_block);
    dim3 block_rows(threads);


    sat_row_prefix_warp << <grid_rows, block_rows >> > (d, d_sat, rows, cols);
    CUDA_CHECK(cudaGetLastError());

    // 3. SAT Col Prefix (STANDARD - Already Optimized)
    {
        const int BX = 128;
        dim3 block(BX);
        dim3 grid_cols((cols + BX - 1) / BX);

        sat_col_prefix << <grid_cols, block >> > (d_sat, rows, cols);
        CUDA_CHECK(cudaGetLastError());
    }

    // 4. Detect (OPTIMIZED - Texture Cache)

    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x,
        (rows + block.y - 1) / block.y);


    cfar_detect_sat << <grid, block >> > (
        d_sat,d,
        b,
        rows, cols,
        win_r, win_c,
        guard_r, guard_c,
        Nref, alpha);
    CUDA_CHECK(cudaGetLastError());

}

void GPUCFAR_SAT::destroy() 
{
    if (d_sat) { cudaFree(d_sat);    d_sat = nullptr; }
}