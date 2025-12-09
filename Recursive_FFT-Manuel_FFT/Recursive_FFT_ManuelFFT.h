#pragma once
#include "../defines.h"


class Recursive_FFT_Manuel_FFT
{
public:
    Recursive_FFT_Manuel_FFT();
    ~Recursive_FFT_Manuel_FFT();

    void run_gpu_pipeline(const std::vector<Complex>& h_input, std::vector<Complex>& h_output);
    void run_cpu_pipeline(const std::vector<Complex>& input, std::vector<Complex>& output);
    
    float getCpuTime();
    float getGPUTime();
    std::vector<Complex> getCpuResult();
    std::vector<Complex> getGpuResult();

private:        
    std::vector<Complex> cpuResult, gpuResult;
    float cpuTime, gpuTime;
    
    // GPU Pointerları
    cuComplex *d_data;
    cuComplex *d_transposed;
    
    cudaEvent_t start, stop; 
    
    int log2_samples; 
    int log2_chirps;  

    void cpu_recursive_fft(std::vector<Complex>& a);
    
    // DÜZELTME: 'temp_ptr' argümanı eklendi
    void execute_naive_fft(cuComplex* data_ptr, cuComplex* temp_ptr, int n, int batch_count, int log2_n);
};

