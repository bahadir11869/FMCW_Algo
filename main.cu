// nvcc main.cu --options-file compile.txt;./myapp.exe

#include "Recursive_FFT-Manuel_Transpose_Shared_Mem/Recursive_FFT.h"
#include "OpenMP-Manuel_Transpose_Shared_Mem/Recursive_FFT_OMP.h"
#include "AVX-Manuel_Transpose_Shared_Mem/AVX_FFT_OMP.h"
#include "AVX-2DFFT/AVX_FFT_OpenMP_2DFFT.h"
#include "defines.h"

std::vector<stTarget> s1 = { 
        {1, 40.0f,  15.0f, 1.0f},
        {2, 85.5f, -10.0f, 0.5f},
        {3, 20.0f,   5.0f, 0.8f}
        };

int main()
{
    Recursive_FFT RecursiveFFT_ManuelTranspose;
    Recursive_FFT_OpenMP RecursiveFFT_ManuelTranspose_OpenMP;
    AVX_FFT_OpenMP AVX_FFT_OpenMP_ManuelTranspose;
    AVX_FFT_OpenMP_2DFFT AVX_FFT_OpenMP_OtoTranspose;
    
    std::vector<Complex> inputData(TOTAL_SIZE);
    std::vector<Complex> outputDataCpu(TOTAL_SIZE);
    std::vector<Complex> outputDataGpu(TOTAL_SIZE);
    veriUret(inputData, s1);        
    int iTekrarSayisi = 5;
    float fCpuTime = 0.0;
    float fGpuTime = 0.0;

    // Recursive_FFT-Manuel_Transpose_Shared_Mem
    for(int i = 0; i <iTekrarSayisi; i++)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
        RecursiveFFT_ManuelTranspose.run_cpu_pipeline(inputData, outputDataCpu);
        fCpuTime += RecursiveFFT_ManuelTranspose.getCpuTime();
    }

    for(int i = 0; i <iTekrarSayisi; i++)
    {
        gpuErrchk(cudaDeviceSynchronize());
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
        RecursiveFFT_ManuelTranspose.run_gpu_pipeline(inputData, outputDataGpu);
        fGpuTime += RecursiveFFT_ManuelTranspose.getGPUTime();
    }
    
    dosyayaYaz("Recursive_FFT-Manuel_Transpose_Shared_Mem/cpu", fCpuTime / (float)iTekrarSayisi,  fGpuTime / (float)iTekrarSayisi, outputDataCpu, s1, 100.0);
    dosyayaYaz("Recursive_FFT-Manuel_Transpose_Shared_Mem/gpu", fCpuTime / (float)iTekrarSayisi,  fGpuTime / (float)iTekrarSayisi, outputDataGpu, s1, 100.0);

    printf("ortalama cpu suresi : %f ms ortalama gpu suresi %f ms\n", fCpuTime / (float)iTekrarSayisi,  fGpuTime / (float)iTekrarSayisi);
    fCpuTime = 0.0;
    fGpuTime = 0.0;

    // OpenMP-Manuel_Transpose_Shared_Mem
    for(int i = 0; i <iTekrarSayisi; i++)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
        RecursiveFFT_ManuelTranspose_OpenMP.run_cpu_pipeline(inputData, outputDataCpu);
        fCpuTime += RecursiveFFT_ManuelTranspose_OpenMP.getCpuTime();
    }

    for(int i = 0; i <iTekrarSayisi; i++)
    {
        gpuErrchk(cudaDeviceSynchronize());
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
        RecursiveFFT_ManuelTranspose_OpenMP.run_gpu_pipeline(inputData, outputDataGpu);
        fGpuTime += RecursiveFFT_ManuelTranspose_OpenMP.getGPUTime();
    }
    
    dosyayaYaz("OpenMP-Manuel_Transpose_Shared_Mem/cpu", fCpuTime / (float)iTekrarSayisi,  fGpuTime / (float)iTekrarSayisi, outputDataCpu, s1, 100.0);
    dosyayaYaz("OpenMP-Manuel_Transpose_Shared_Mem/gpu", fCpuTime / (float)iTekrarSayisi,  fGpuTime / (float)iTekrarSayisi, outputDataGpu, s1, 100.0);

    printf("ortalama cpu suresi : %f ms ortalama gpu suresi %f ms\n", fCpuTime / (float)iTekrarSayisi,  fGpuTime / (float)iTekrarSayisi);
    gpuErrchk(cudaDeviceSynchronize());
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    fCpuTime = 0.0;
    fGpuTime = 0.0;
    // AVX-Manuel_Transpose_Shared_Mem
    for(int i = 0; i <iTekrarSayisi; i++)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
        AVX_FFT_OpenMP_ManuelTranspose.run_cpu_pipeline(inputData, outputDataCpu);
        fCpuTime += AVX_FFT_OpenMP_ManuelTranspose.getCpuTime();
    }

    for(int i = 0; i <iTekrarSayisi; i++)
    {
        gpuErrchk(cudaDeviceSynchronize());
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
        AVX_FFT_OpenMP_ManuelTranspose.run_gpu_pipeline(inputData, outputDataGpu);
        fGpuTime += AVX_FFT_OpenMP_ManuelTranspose.getGPUTime();
    }
    
    dosyayaYaz("AVX-Manuel_Transpose_Shared_Mem/cpu", fCpuTime / (float)iTekrarSayisi,  fGpuTime / (float)iTekrarSayisi, outputDataCpu, s1, 100.0);
    dosyayaYaz("AVX-Manuel_Transpose_Shared_Mem/gpu", fCpuTime / (float)iTekrarSayisi,  fGpuTime / (float)iTekrarSayisi, outputDataGpu, s1, 100.0);

    printf("ortalama cpu suresi : %f ms ortalama gpu suresi %f ms\n", fCpuTime / (float)iTekrarSayisi,  fGpuTime / (float)iTekrarSayisi);
    gpuErrchk(cudaDeviceSynchronize());
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    fCpuTime = 0.0;
    fGpuTime = 0.0;

    // AVX-2DFFT
    Complex *h_pinned_input, *h_pinned_outputGPU, *h_pinned_outputCPU;
    gpuErrchk(cudaMallocHost((void**)&h_pinned_input, TOTAL_SIZE * sizeof(Complex)));
    gpuErrchk(cudaMallocHost((void**)&h_pinned_outputCPU, TOTAL_SIZE * sizeof(Complex)));
    gpuErrchk(cudaMallocHost((void**)&h_pinned_outputGPU, TOTAL_SIZE * sizeof(Complex)));
    memcpy(h_pinned_input, inputData.data(), TOTAL_SIZE* sizeof(Complex));
    for(int i = 0; i <iTekrarSayisi; i++)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
        AVX_FFT_OpenMP_OtoTranspose.run_cpu_pipeline(h_pinned_input, h_pinned_outputCPU);
        fCpuTime += AVX_FFT_OpenMP_OtoTranspose.getCpuTime();
    }

    for(int i = 0; i <iTekrarSayisi; i++)
    {
        gpuErrchk(cudaDeviceSynchronize());
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
        AVX_FFT_OpenMP_OtoTranspose.run_gpu_pipeline(h_pinned_input, h_pinned_outputGPU);
        fGpuTime += AVX_FFT_OpenMP_OtoTranspose.getGPUTime();
    }
    
    memcpy(outputDataGpu.data(), h_pinned_outputGPU, sizeof(Complex) * TOTAL_SIZE);
    memcpy(outputDataCpu.data(), h_pinned_outputCPU, sizeof(Complex) * TOTAL_SIZE);
    dosyayaYaz("AVX-2DFFT/cpu", fCpuTime / (float)iTekrarSayisi,  fGpuTime / (float)iTekrarSayisi, outputDataCpu, s1, 100.0, false);
    dosyayaYaz("AVX-2DFFT/gpu", fCpuTime / (float)iTekrarSayisi,  fGpuTime / (float)iTekrarSayisi, outputDataGpu, s1, 100.0, true);

    printf("ortalama cpu suresi : %f ms ortalama gpu suresi %f ms\n", fCpuTime / (float)iTekrarSayisi,  fGpuTime / (float)iTekrarSayisi);

    return 0;
}