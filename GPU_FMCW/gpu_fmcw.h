#pragma once
#include "../defines.h"

class gpu_fmcw
{
private:
    float fgpuTime;
    float fgpuComputeTime;
    cuComplex *d_data;
    cuComplex *d_transposed;
    cudaEvent_t start, stop; 
    cudaEvent_t start2, stop2; 
    cufftHandle plan;
    cufftHandle planRange;
    cufftHandle planDoppler;
    int fftType;
    int log2_samples; 
    int log2_chirps;  
    void execute_naive_fft(cuComplex* data_ptr, cuComplex* temp_ptr, int n, int batch_count, int log2_n);

public:
    gpu_fmcw(int fftType);
    ~gpu_fmcw();
    void run_gpu_manuel_transpose(std::vector<Complex>& input, std::vector<Complex>& output);
    void run_gpu_manuel_FFT_Shared_Yok(std::vector<Complex>& input, std::vector<Complex>& output);
    void run_gpu_manuel_FFT_Shared_Mem(std::vector<Complex>& input, std::vector<Complex>& output);
    void run_gpu_2DFFT(Complex* input, Complex* output);

    float getGpuTime();
    float getGpuComputeTime();
};
