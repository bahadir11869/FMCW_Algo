#pragma once
#include "../defines.h"

class Recursive_FFT_OpenMP
{
    public:
        void run_gpu_pipeline(const std::vector<Complex> h_input, std::vector<Complex>& h_output);
        void run_cpu_pipeline(const std::vector<Complex> input, std::vector<Complex>& output);
        float getCpuTime();
        float getGPUTime();
        std::vector<Complex> getCpuResult();
        std::vector<Complex> getGpuResult();

    private:        
        std::vector<Complex> cpuResult, gpuResult;
        void cpu_recursive_fft(std::vector<Complex>& a);
        float cpuTime, gpuTime;

}; 