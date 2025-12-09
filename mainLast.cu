#include "defines.h"
#include "GPU_FMCW/gpu_fmcw.h"
#include "CPU_FMCW/cpu_fmcw.h"

std::vector<stTarget> s1 = { 
        {1, 40.0f,  15.0f, 1.0f},
        {2, 70.0f, -10.0f, 0.5f},
        {3, 20.0f,   5.0f, 0.8f}
        };

int main()
{
    std::vector<Complex> inputData(TOTAL_SIZE);

    std::vector<Complex> outputDataCpu(TOTAL_SIZE);
    std::vector<Complex> outputDataCpuOpenMP(TOTAL_SIZE);
    std::vector<Complex> outputDataCpuAVX(TOTAL_SIZE);

    std::vector<Complex> outputDataGpuManuel(TOTAL_SIZE);
    std::vector<Complex> outputDataGpuManuelTranspose(TOTAL_SIZE);
    std::vector<Complex> outputDataGpu2D(TOTAL_SIZE);


    veriUret(inputData, s1);   

    gpu_fmcw* gpuFMCW = new gpu_fmcw(1);
    gpu_fmcw* gpuFMCW2 = new gpu_fmcw(2);

    cpu_fmcw* cpuFMCW = new cpu_fmcw();

    int iTekrarSayisi = 5;

    float fCpuTime = 0.0;
    float fCpuTimeOpenMP = 0.0;
    float fCpuTimeAVX = 0.0;
    float fCpuComputeTimeAVX = 0.0;
    
    float fGpuTime = 0.0;
    float fGpuComputeTime = 0.0;

    float fGpuTimeManuel = 0.0;
    float fGpuComputeTimeManuel = 0.0;
    
    float fGpuTime2D = 0.0;
    float fGpuComputeTime2D = 0.0;
    
    const char* dosyaAdi = "";

    for(int i = 0; i < iTekrarSayisi; i++)
    {
        gpuFMCW->run_gpu_manuel_FFT_Shared_Yok(inputData, outputDataGpuManuel);
        gpuErrchk(cudaDeviceSynchronize());
        fGpuTimeManuel += gpuFMCW->getGpuTime();
        fGpuComputeTimeManuel += gpuFMCW->getGpuComputeTime();
    }

    dosyaAdi = "GPU_FMCW/gpu_Manuel_FFT_SharedYok.txt";
    fs::remove(dosyaAdi);
    dosyayaYaz(dosyaAdi, fCpuTime / (float)iTekrarSayisi,  fGpuTime2D / (float)iTekrarSayisi, outputDataGpuManuel, s1, 100.0);

    printf("GPU Manuel FFT Shared yok: Compute time %f total time %f \n", float(fGpuComputeTimeManuel/iTekrarSayisi), float(fGpuTimeManuel/iTekrarSayisi));
    fGpuTimeManuel = 0.0;
    fGpuComputeTimeManuel = 0.0;
    for(int i = 0; i < iTekrarSayisi; i++)
    {
        gpuFMCW->run_gpu_manuel_FFT_Shared_Mem(inputData, outputDataGpuManuel);
        gpuErrchk(cudaDeviceSynchronize());
        fGpuTimeManuel += gpuFMCW->getGpuTime();
        fGpuComputeTimeManuel += gpuFMCW->getGpuComputeTime();
    }
 
    printf("GPU Manuel FFT Shared Mem: Compute time %f total time %f \n", float(fGpuComputeTimeManuel/iTekrarSayisi), float(fGpuTimeManuel/iTekrarSayisi));

    for(int i = 0; i < iTekrarSayisi; i++)
    {
        gpuFMCW->run_gpu_manuel_transpose(inputData, outputDataGpuManuelTranspose);
        gpuErrchk(cudaDeviceSynchronize());
        fGpuTime += gpuFMCW->getGpuTime();
        fGpuComputeTime += gpuFMCW->getGpuComputeTime();
    }

    printf("GPU Manuel Transpose: Compute time %f total time %f \n", float(fGpuComputeTime/iTekrarSayisi), float(fGpuTime/iTekrarSayisi));

    Complex *h_pinned_input, *h_pinned_outputGPU, *h_pinned_outputCPU;
    gpuErrchk(cudaMallocHost((void**)&h_pinned_input, TOTAL_SIZE * sizeof(Complex)));
    gpuErrchk(cudaMallocHost((void**)&h_pinned_outputCPU, TOTAL_SIZE * sizeof(Complex)));
    gpuErrchk(cudaMallocHost((void**)&h_pinned_outputGPU, TOTAL_SIZE * sizeof(Complex)));
    memcpy(h_pinned_input, inputData.data(), TOTAL_SIZE* sizeof(Complex));

    for(int i = 0; i < iTekrarSayisi; i++)
    {
        gpuFMCW2->run_gpu_2DFFT(h_pinned_input, h_pinned_outputGPU);
        gpuErrchk(cudaDeviceSynchronize());
        fGpuTime2D += gpuFMCW2->getGpuTime();
        fGpuComputeTime2D += gpuFMCW2->getGpuComputeTime();
    }
    memcpy(outputDataGpu2D.data(), h_pinned_outputGPU, sizeof(Complex) * TOTAL_SIZE);
    printf("2D_FFT Compute time %f total time %f \n", float(fGpuComputeTime2D/iTekrarSayisi), float(fGpuTime2D/iTekrarSayisi));


    for(int i = 0; i < iTekrarSayisi; i++)
    {
        cpuFMCW->run_cpu_basic(inputData, outputDataCpu); 
        fCpuTime += cpuFMCW->getCpuTime();

        cpuFMCW->run_cpu_openmp(inputData, outputDataCpuOpenMP);
        fCpuTimeOpenMP += cpuFMCW->getCpuTime();
        
        cpuFMCW->run_cpu_avx(h_pinned_input, h_pinned_outputCPU);
        fCpuComputeTimeAVX += cpuFMCW->getCpuComputeTime();
        fCpuTimeAVX += cpuFMCW->getCpuTime();
        
        
    }

    memcpy(outputDataCpuAVX.data(), h_pinned_outputCPU, sizeof(Complex) * TOTAL_SIZE);
    printf("CPU Recursive FFT Time : %f CPU Recurisive FFT OpenMP Time: %f CPU AVX Total Time: %f CPU AVX Compute Time: %f\n", 
        float(fCpuTime/iTekrarSayisi), float(fCpuTimeOpenMP/iTekrarSayisi), float(fCpuTimeAVX/iTekrarSayisi), float(fCpuComputeTimeAVX/iTekrarSayisi));

    delete gpuFMCW;
    delete gpuFMCW2;
    delete cpuFMCW;

    cudaFreeHost(h_pinned_input);
    cudaFreeHost(h_pinned_outputCPU);
    cudaFreeHost(h_pinned_outputGPU);

    dosyaAdi = "GPU_FMCW/gpu_Manuel_FFT.txt";
    fs::remove(dosyaAdi);
    dosyayaYaz(dosyaAdi, fCpuTime / (float)iTekrarSayisi,  fGpuTime2D / (float)iTekrarSayisi, outputDataGpuManuel, s1, 100.0);
    
    dosyaAdi = "GPU_FMCW/gpu_2D.txt";
    fs::remove(dosyaAdi);
    dosyayaYaz(dosyaAdi, fCpuTime / (float)iTekrarSayisi,  fGpuTime2D / (float)iTekrarSayisi, outputDataGpu2D, s1, 100.0, true);

    dosyaAdi = "GPU_FMCW/gpu_ManuelTranspose.txt";
    fs::remove(dosyaAdi);
    dosyayaYaz(dosyaAdi, fCpuTime / (float)iTekrarSayisi,  fGpuTime / (float)iTekrarSayisi, outputDataGpuManuelTranspose, s1, 100.0);

    dosyaAdi = "CPU_FMCW/cpu_Recursive.txt";
    fs::remove(dosyaAdi);
    dosyayaYaz(dosyaAdi, fCpuTime / (float)iTekrarSayisi,  fGpuTime / (float)iTekrarSayisi, outputDataCpu, s1, 100.0);

    dosyaAdi = "CPU_FMCW/cpu_Recursive_OpenMP.txt";
    fs::remove(dosyaAdi);
    dosyayaYaz(dosyaAdi, fCpuTime / (float)iTekrarSayisi,  fGpuTime / (float)iTekrarSayisi, outputDataCpuOpenMP, s1, 100.0);

    dosyaAdi = "CPU_FMCW/cpu_AVX.txt";
    fs::remove(dosyaAdi);
    dosyayaYaz(dosyaAdi, fCpuTime / (float)iTekrarSayisi,  fGpuTime / (float)iTekrarSayisi, outputDataCpuAVX, s1, 100.0);

    /*    
    dosyaAdi = "GPU_FMCW/gpu_2D.txt";
    fs::remove(dosyaAdi);
    dosyayaYaz(dosyaAdi, fCpuTime / (float)iTekrarSayisi,  fGpuTime / (float)iTekrarSayisi, outputDataGpu2D, s1, 100.0);
    
    dosyaAdi = "GPU_FMCW/gpu_2D.txt";
    fs::remove(dosyaAdi);
    dosyayaYaz(dosyaAdi, fCpuTime / (float)iTekrarSayisi,  fGpuTime / (float)iTekrarSayisi, outputDataGpu2D, s1, 100.0);
    */

    
}