// nvcc -arch=sm_86 -std=c++17 -O3  mainLast.cu --options-file compile2.txt;.\FMCW_Algo2.exe
#include "defines.h"
#include "GPU_FMCW/gpu_fmcw.h"
#include "CPU_FMCW/cpu_fmcw.h"

std::vector<stTarget> s1 = { 
        {1, 40.0f,  15.0f, 1.0f},
        {2, 60.0f, -10.0f, 0.5f},
        {3, 20.0f,   5.0f, 0.8f}
        };



int main()
{
    std::vector<Complex> fullData(TOTAL_ELEMENTS);
    Complex *h_pinned_input, *h_pinned_outputCPU;
    float* h_pinned_outputGPU;
    gpuErrchk(cudaMallocHost((void**)&h_pinned_input, TOTAL_ELEMENTS * sizeof(Complex)));
    gpuErrchk(cudaMallocHost((void**)&h_pinned_outputCPU, TOTAL_SIZE * sizeof(Complex)));
    gpuErrchk(cudaMallocHost((void**)&h_pinned_outputGPU, TOTAL_SIZE * sizeof(float)));
    std::vector<float> detectionMap(TOTAL_SIZE);

    readBin("radar_raw_frameBin/000357.bin", fullData);
    //veriUret(fullData, s1);

    int iTekrarSayisi = 10;

    gpu_fmcw gpuManuelSHM(1,"GPU_FMCW/gpu_Manuel_FFT_Shared.txt");
    for(int i = 0; i < iTekrarSayisi; i++)
    {
        gpuManuelSHM.run_gpu_manuel_FFT_Shared_Mem(fullData, h_pinned_outputGPU);        
        gpuErrchk(cudaDeviceSynchronize());
    }


    memcpy(h_pinned_input, fullData.data(), TOTAL_ELEMENTS * sizeof(Complex));
    gpu_fmcw gpu2DFFT(2,"GPU_FMCW/2DFFT.txt");

    for(int i = 0; i < iTekrarSayisi; i++)
    {
        gpu2DFFT.run_gpu_2DFFT(h_pinned_input, h_pinned_outputGPU);
        gpuErrchk(cudaDeviceSynchronize());
    }
    applyPeakRelativeFilter(gpu2DFFT.bpCFAR, gpu2DFFT.getOutput(), NUM_CHIRPS, NUM_SAMPLES);
    for(int i=0; i<TOTAL_SIZE; ++i) {
    detectionMap[i] = (gpu2DFFT.bpCFAR[i]) ? 1.0f : 0.0f;
    }

    save_rdm_data("GPU_FMCW/CFAR_2DFFT.csv", detectionMap.data(), true);
    save_rdm_data("GPU_FMCW/FFT_2DFFT.csv", gpu2DFFT.getOutput(), true);

    applyPeakRelativeFilter(gpuManuelSHM.bpCFAR, gpuManuelSHM.getOutput(), NUM_CHIRPS, NUM_SAMPLES);
    for(int i=0; i<TOTAL_SIZE; ++i) {
    detectionMap[i] = (gpuManuelSHM.bpCFAR[i]) ? 1.0f : 0.0f;
    }
    save_rdm_data("GPU_FMCW/CFAR_ManuelSHM.csv", detectionMap.data(), true);
    save_rdm_data("GPU_FMCW/FFT_ManuelSHM.csv", gpuManuelSHM.getOutput(), true);


    cpu_fmcw cpuFMCWManuel("CPU_FMCW/cpu_Recursive.txt");
    cpuFMCWManuel.run_cpu_basic(fullData);                


    for(int i=0; i<TOTAL_SIZE; ++i) {
    detectionMap[i] = (cpuFMCWManuel.cfarData.truth[i]) ? 1.0f : 0.0f;
    }
    save_rdm_data("CPU_FMCW/CFAR_CPU.csv", detectionMap.data(), true);
    save_rdm_data("CPU_FMCW/FFT_CPU.csv", cpuFMCWManuel.getOutput(), true);

    
    cpu_fmcw cpuFMCWAVX("CPU_FMCW/cpu_AVX.txt");
    memcpy(h_pinned_input, fullData.data(), TOTAL_ELEMENTS * sizeof(Complex));
    cpuFMCWAVX.run_cpu_avx(h_pinned_input, h_pinned_outputCPU);
    


    for(int i=0; i<TOTAL_SIZE; ++i) 
    {
        detectionMap[i] = (cpuFMCWAVX.cfarData.truth[i]) ? 1.0f : 0.0f;
    }

    save_rdm_data("CPU_FMCW/CFAR_AVX.csv", detectionMap.data(), true);
    save_rdm_data("CPU_FMCW/FFT_AVX.csv", cpuFMCWAVX.getOutput(), true);
    

    printf("CPU Recurisive FFT OpenMP  FMCW time: %f ms, CFAR time: %f ms, TOTAL time: %f ms\n", cpuFMCWManuel.getCpuTime(), cpuFMCWManuel.getCfarCpuTime(), cpuFMCWManuel.getCpuTime() + cpuFMCWManuel.getCfarCpuTime());
    fflush(stdout);
    printf("CPU AVX FFT OpenMP         FMCW time: %f ms, CFAR time: %f ms, TOTAL time: %f ms\n", cpuFMCWAVX.getCpuTime(), cpuFMCWAVX.getCfarCpuTime(), cpuFMCWAVX.getCpuTime() + cpuFMCWAVX.getCfarCpuTime());
    fflush(stdout);
    printf("GPU Manuel FFT Shared Mem: FMCW time: %f ms, CFAR time: %f ms, TOTAL time: %f ms \n", gpuManuelSHM.getGpuComputeTime(), gpuManuelSHM.getCfarGpuComputeTime(), gpuManuelSHM.getGpuTime());    
    fflush(stdout);
    printf("2D_FFT                     FMCW time: %f ms, CFAR time: %f ms, TOTAL time: %f ms \n", gpu2DFFT.getGpuComputeTime(), gpu2DFFT.getCfarGpuComputeTime(), gpu2DFFT.getGpuTime());    
    fflush(stdout);
    cudaFreeHost(h_pinned_input);
    cudaFreeHost(h_pinned_outputCPU);
    cudaFreeHost(h_pinned_outputGPU);
    return 0;
}