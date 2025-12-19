#pragma once
#include "../defines.h"

class gpu_fmcw
{
private:
    
    float fgpuTime;
    float fgpuComputeTime;
    
    std::string strDosyaAdi;

    std::vector<float> vfgpuTime;
    std::vector<float> vfgpuComputeTime;
    std::vector<Complex> output;
    
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

    cudaStream_t streams[4]; // 4 tane paralel işçi tanımlıyoruz

    void execute_naive_fft(cuComplex* data_ptr, cuComplex* temp_ptr, int n, int batch_count, int log2_n);

public:

    gpu_fmcw(int fftType, std::string strDosyaAdi);
    ~gpu_fmcw();
    void run_gpu_manuel_transpose(std::vector<Complex>& input);
    void run_gpu_manuel_FFT_Shared_Yok(std::vector<Complex>& input);
    void run_gpu_manuel_FFT_Shared_Mem(std::vector<Complex>& input);
    void run_gpu_2DFFT(Complex* input, Complex* ptroutput);
    void run_gpu_streams(Complex* h_input, Complex* h_output);

    float getGpuTime();
    float getGpuComputeTime();
    std::string getDosyaAdi();
    std::vector<Complex> getOutput();


};
