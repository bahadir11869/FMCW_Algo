#pragma once
#include "../defines.h"

class cpu_fmcw
{
private:
    void cpu_recursive_fft(std::vector<Complex>& a); 
    float fcpuTime;
    float fcpuAVXComputeTime;

public:
    cpu_fmcw();
    ~cpu_fmcw();
    void run_cpu_basic(const std::vector<Complex>& input, std::vector<Complex>& output);
    void run_cpu_openmp(const std::vector<Complex>& input, std::vector<Complex>& output);
    void run_cpu_avx(Complex* input, Complex* output);

    float getCpuTime();
    float getCpuComputeTime();
};

