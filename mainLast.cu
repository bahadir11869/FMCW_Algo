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
    long long transferred_bytes = TOTAL_SIZE * sizeof(float) + TOTAL_ELEMENTS * sizeof(float);
    std::vector<float> detectionMap(TOTAL_SIZE);

    readBin("radar_raw_frameBin/000006.bin", fullData);
    

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
    for(int i=0; i<TOTAL_SIZE; ++i) {
    detectionMap[i] = (gpu2DFFT.bpCFAR[i]) ? 1.0f : 0.0f;
    }
    
    save_rdm_data("GPU_FMCW/CFAR_2DFFT.csv", detectionMap.data(), true);
    save_rdm_data("GPU_FMCW/FFT_2DFFT.csv", gpu2DFFT.getOutput(), true);

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
    
    printf("\nCPU AVX Total time: %f\n", cpuFMCWAVX.getCpuTime());


    for(int i=0; i<TOTAL_SIZE; ++i) 
    {
        detectionMap[i] = (cpuFMCWAVX.cfarData.truth[i]) ? 1.0f : 0.0f;
    }

    save_rdm_data("CPU_FMCW/CFAR_AVX.csv", detectionMap.data(), true);
    save_rdm_data("CPU_FMCW/FFT_AVX.csv", cpuFMCWAVX.getOutput(), true);

    printf("GPU Manuel FFT Shared Mem: Compute time: %f ms  total time: %f ms bandWithGPUManuelSharedMem : %f GB/s RTX 3060 Max BandWith: ~360 GB/s \n", gpuManuelSHM.getGpuComputeTime(),  gpuManuelSHM.getGpuTime(), (transferred_bytes * 1e-9)/(gpuManuelSHM.getGpuComputeTime()/1000.0));    
    printf("2D_FFT Compute time: %f ms total time: %f ms bandWithGPU2D : %f GB/s RTX 3060 Max BandWith: ~360 GB/s\n", gpu2DFFT.getGpuComputeTime(),gpu2DFFT.getGpuTime(), (transferred_bytes * 1e-9)/(gpu2DFFT.getGpuComputeTime()/1000.0));
    printf("CPU Recurisive FFT OpenMP time: %f\n", cpuFMCWManuel.getCpuTime());

    /*
    memcpy(h_pinned_input, fullData.data(), TOTAL_ELEMENTS * sizeof(Complex));
    gpu_fmcw gpu2DFFT(2,"GPU_FMCW/2DFFT.txt");

    for(int i = 0; i < iTekrarSayisi; i++)
    {
        gpu2DFFT.run_gpu_2DFFT(h_pinned_input, h_pinned_outputGPU);
        gpuErrchk(cudaDeviceSynchronize());
    }


    FILE* file14 = fopen("GPU_FMCW/fft2D.txt", "w+");
    for (int i = 0; i < 2000; ++i) {
        fprintf(file14, "genlik:  %f\n", gpu2DFFT.getOutput()[i]);
    }
    fclose(file14);

    cpu_fmcw cpuFMCWAVX("CPU_FMCW/cpu_AVX.txt");
    memcpy(h_pinned_input, fullData.data(), TOTAL_ELEMENTS * sizeof(Complex));
    cpuFMCWAVX.run_cpu_avx(h_pinned_input, h_pinned_outputCPU);
        
    FILE* file15 = fopen("CPU_FMCW/avxFFT.txt", "w+");
    for (int i = 0; i < 2000; ++i) {
        fprintf(file15, "genlik:  %f\n", cpuFMCWAVX.getOutput()[i]);
    }
    fclose(file15);
    */

    /* duz fmcw ve cfar eklendi.
    cpu_fmcw cpuFMCWManuel("CPU_FMCW/cpu_Recursive.txt");
    cpuFMCWManuel.run_cpu_basic(fullData);                

    FILE* file13 = fopen("CPU_FMCW/duzFFT.txt", "w+");
    printf("CPU Basic %f \n", cpuFMCWManuel.getOutput()[1]);
    for (int i = 0; i < TOTAL_SIZE; ++i) {
        fprintf(file13, "genlik:  %f\n", cpuFMCWManuel.getOutput()[i]);
    }
    fclose(file13);
    std::vector<float> detectionMap(TOTAL_SIZE);
    for(int i=0; i<TOTAL_SIZE; ++i) {
    detectionMap[i] = (cpuFMCWManuel.cfarData.truth[i]) ? 1.0f : 0.0f;
    }
    save_rdm_data("CPU_FMCW/radar_mask_duz_FMCW.csv", detectionMap.data(), true);
    save_rdm_data("CPU_FMCW/duz_FMCW.csv", cpuFMCWManuel.getOutput(), true);
    */




    //printf("GPU Manuel FFT Shared Mem: Compute time: %f ms  total time: %f ms bandWithGPUManuelSharedMem : %f GB/s RTX 3060 Max BandWith: ~360 GB/s \n", gpuManuelSHM.getGpuComputeTime(),  gpuManuelSHM.getGpuTime(), (transferred_bytes * 1e-9)/(gpuManuelSHM.getGpuComputeTime()/1000.0));    
    //printf("2D_FFT Compute time: %f ms total time: %f ms bandWithGPU2D : %f GB/s RTX 3060 Max BandWith: ~360 GB/s\n", gpu2DFFT.getGpuComputeTime(),gpu2DFFT.getGpuTime(), (transferred_bytes * 1e-9)/(gpu2DFFT.getGpuComputeTime()/1000.0));
    
    //printf("\n\n\t\t\t\t\t\t\t\t\t --------SONUCLAR CPU --------- \n\n");
    //printf("CPU Recurisive FFT OpenMP time: %f CPU AVX Total time: %f\n", 
    //cpuFMCWManuel.getCpuTime(), cpuFMCWAVX.getCpuTime());
    
    //save_rdm_data("GPU_FMCW/2DFFT_GPU.csv", gpu2DFFT.getOutput(), true);
    //save_rdm_data("CPU_FMCW/AVXFFT_CPU.csv", cpuFMCWAVX.getOutput(), true);  
    
    /*
    save_rdm_data("GPU_FMCW/2DFFT_GPU.csv", gpu2DFFT.getOutput(), true);
    save_rdm_data("CPU_FMCW/AVXFFT_CPU.csv", cpuFMCWAVX.getOutput(), true);  
      
    printf("\n\n\t\t\t\t\t\t\t\t\t --------SONUCLAR GPU --------- \n\n");
    printf("GPU Manuel FFT Shared Mem: Compute time: %f ms  total time: %f ms bandWithGPUManuelSharedMem : %f GB/s RTX 3060 Max BandWith: ~360 GB/s \n", gpuManuelSHM.getGpuComputeTime(),  gpuManuelSHM.getGpuTime(), (transferred_bytes * 1e-9)/(gpuManuelSHM.getGpuComputeTime()/1000.0));    
    printf("2D_FFT Compute time: %f ms total time: %f ms bandWithGPU2D : %f GB/s RTX 3060 Max BandWith: ~360 GB/s\n", gpu2DFFT.getGpuComputeTime(),gpu2DFFT.getGpuTime(), (transferred_bytes * 1e-9)/(gpu2DFFT.getGpuComputeTime()/1000.0));
    
    printf("\n\n\t\t\t\t\t\t\t\t\t --------SONUCLAR CPU --------- \n\n");
    printf("CPU Recurisive FFT OpenMP time: %f CPU AVX Total time: %f\n", 
    cpuFMCWManuel.getCpuTime(), cpuFMCWAVX.getCpuTime());
    
    printf("\n\n\t\t\t\t\t\t\t\t\t --------SONUCLAR CPU AVX vs GPU  --------- \n\n");

    
    

    printf("GPUManuelFFTSharedMem vs AVX RMSE \n");
    calculate_RMSE_Vectors(cpuFMCWAVX.getOutput(), gpuManuelSHM.getOutput(), NUM_SAMPLES, NUM_CHIRPS, false);  

    printf("Gpu2D vs AVX RMSE \n");
    calculate_RMSE_Vectors(cpuFMCWAVX.getOutput(), gpu2DFFT.getOutput(), NUM_SAMPLES, NUM_CHIRPS, false);  

    */   
    cudaFreeHost(h_pinned_input);
    cudaFreeHost(h_pinned_outputCPU);
    cudaFreeHost(h_pinned_outputGPU);
}