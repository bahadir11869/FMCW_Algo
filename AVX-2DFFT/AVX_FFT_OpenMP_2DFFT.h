#pragma once
#include "../defines.h"

class AVX_FFT_OpenMP_2DFFT
{
    public:
        AVX_FFT_OpenMP_2DFFT();
        ~AVX_FFT_OpenMP_2DFFT();
        void run_gpu_pipeline(Complex* h_input, Complex* h_output);
        void run_cpu_pipeline(Complex* input, Complex* output);
        float getCpuTime();
        float getGPUTime();
        std::vector<Complex> getCpuResult();
        std::vector<Complex> getGpuResult();

    private:        
        std::vector<Complex> cpuResult, gpuResult;
        float cpuTime, gpuTime;
        cuComplex* d_data;
        cufftHandle plan;

}; 