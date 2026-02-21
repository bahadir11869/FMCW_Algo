#pragma once
#include "../defines.h"
#include "../GPU_CFAR/cfar_gpu_sat.hpp"

class gpu_fmcw
{
private:
    
    float fgpuTime;
    float fgpuComputeTime;
    float cfarComputeTime;

    std::string strDosyaAdi;

    std::vector<float> vfgpuTime;
    std::vector<float> vfgpuComputeTime;
    std::vector<float> vfCfargpuTime;
    std::vector<Complex> output;
    
    float* f_data;
    float* f_data_host;
    cuComplex *d_data;
    cuComplex *d_data_all;
    cuComplex *d_transposed;
    cuComplex *d_transposed_all;

    cudaEvent_t start_total, stop_total; 
    cudaEvent_t start_fmcw_compute, stop_fmcw_compute; 
    cudaEvent_t start_cfar_compute, stop_cfar_compute; 

    cufftHandle plan;
    cufftHandle planRange;
    cufftHandle planDoppler;

    int fftType;
    int log2_samples; 
    int log2_chirps;  

    cudaStream_t streams[4]; // 4 tane paralel işçi tanımlıyoruz

    void execute_naive_fft(cuComplex* data_ptr, cuComplex* temp_ptr, int n, int batch_count, int log2_n);


public:
    bool* bpCFAR;
    float* fpCfarData;
    bool* bpCfarData;
    CFARData cfarData;
    CFARParams cfarParam;
    GPUCFAR_SAT cfarProcessor;
    gpu_fmcw(int fftType, std::string strDosyaAdi);
    ~gpu_fmcw();
    void run_gpu_manuel_transpose(std::vector<Complex>& input);
    void run_gpu_manuel_FFT_Shared_Yok(std::vector<Complex>& input);
    void run_gpu_manuel_FFT_Shared_Mem(std::vector<Complex>& input, float* fOutput);
    void run_gpu_2DFFT(Complex* input, float* ptroutput);
    void run_gpu_streams(Complex* h_input, Complex* h_output);

    float getGpuTime();
    float getGpuComputeTime();
    float getCfarGpuComputeTime();
    std::string getDosyaAdi();
    float* getOutput();


};
