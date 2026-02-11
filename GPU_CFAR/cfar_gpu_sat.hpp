// cfar_gpu_sat.hpp
#pragma once
#include "../defines.h"
#include <cuda_runtime.h>

class GPUCFAR_SAT {
public:
    GPUCFAR_SAT() = default;
    ~GPUCFAR_SAT() { destroy(); }

    void init(const CFARParams& p);
    void process(const CFARParams& p, float* d, bool* b);
    void destroy();

private:
    // Parametreler
    int rows = 0, cols = 0;
    int ref_r = 0, ref_c = 0;
    int guard_r = 0, guard_c = 0;
    int win_r = 0, win_c = 0;
    int Nref = 0;
    float alpha = 0.0f;

    float* d_sat = nullptr;   // 2D SAT (integral image)


    void ensure_device_alloc(size_t Npix);
};
