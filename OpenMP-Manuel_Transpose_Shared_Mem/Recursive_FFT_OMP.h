#pragma once
#include "../defines.h"

class Recursive_FFT_OpenMP
{
    public:
        void run_gpu_pipeline(const std::vector<Complex>& h_input, std::vector<Complex>& h_output);
        void run_cpu_pipeline(const std::vector<Complex>& input, std::vector<Complex>& output);
        Recursive_FFT_OpenMP();
        ~Recursive_FFT_OpenMP();
        float getCpuTime();
        float getGPUTime();
        std::vector<Complex> getCpuResult();
        std::vector<Complex> getGpuResult();

    private:        
        std::vector<Complex> cpuResult, gpuResult;
        void cpu_recursive_fft(std::vector<Complex>& a);
        float cpuTime, gpuTime;
        cudaEvent_t start, stop; 
        cuComplex *d_data, *d_transposed;
        cufftHandle planRange, planDoppler;

}; 