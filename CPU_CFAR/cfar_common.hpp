#pragma once
#include <vector>
#include <random>
#include <cmath>
#include <cstdint>
#include <string>
#include <algorithm>
#include "../defines.h"
inline size_t idx(size_t r, size_t c, size_t C) { return r * C + c; }

class CFARParams {
public:
    int rows = NUM_SAMPLES, cols = NUM_CHIRPS ;
    float snr_db = 10.0f;             // güç SNR (dB)
    int ref_r = 8, ref_c = 8;
    int guard_r = 2, guard_c = 2;
    float pfa = 1e-6f;
};

class CFARData {
public:
    std::vector<float>          power;  // rows*cols
    bool*  truth;  // 0/1
};

class CFARStats {
public:
    // CPU zamanları (ms)
    double cpu_ms_sat = 0.0;
    double cpu_ms_detect = 0.0;
    double cpu_ms_total = 0.0;

    // GPU zamanları (ms)
    float  gpu_ms_total = 0.0f;  // H2D + Kernels + D2H
    float  gpu_ms_h2d = 0.0f;
    float  gpu_ms_kernels = 0.0f;  // alt kalemlerin toplamı
    float  gpu_ms_row_big = 0.0f;
    float  gpu_ms_col_big = 0.0f;
    float  gpu_ms_row_guard = 0.0f;
    float  gpu_ms_col_guard = 0.0f;
    float  gpu_ms_detect = 0.0f;
    float  gpu_ms_d2h = 0.0f;

    // metrikler
    double Pd = 0.0, Pfa_emp = 0.0;

    // çağrı noktası duvar saati
};
