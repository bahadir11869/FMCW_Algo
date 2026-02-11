#pragma once
#include "../defines.h"

class AVXCFAR {
private:
    CFARParams P;
    float alpha; // alpha / N
    int N_ref;
public:
    AVXCFAR(CFARParams params);
    void update_params(CFARParams params);
    // SAT (Summed Area Table) Hesaplaması - Hala OpenMP ile paralel
    void compute_sat(const float* power, float* sat, int rows, int cols);
    // AVX2 OPTİMİZASYONLU DETECT FONKSİYONU
    void process(CFARData& d);
    // Scalar Helper (Cleanup loop için)
    float sat_rect_sum(const float* sat, int rows, int cols, int r, int c, int wr, int wc);
};