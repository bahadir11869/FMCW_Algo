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
    Complex *h_pinned_input, *h_pinned_outputGPU, *h_pinned_outputCPU;
    gpuErrchk(cudaMallocHost((void**)&h_pinned_input, TOTAL_ELEMENTS * sizeof(Complex)));
    gpuErrchk(cudaMallocHost((void**)&h_pinned_outputCPU, TOTAL_SIZE * sizeof(Complex)));
    gpuErrchk(cudaMallocHost((void**)&h_pinned_outputGPU, TOTAL_SIZE * sizeof(Complex)));

    readBin("radar_raw_frameBin/adcData_padded.bin", fullData);
    int iTekrarSayisi = 10;

    gpu_fmcw gpuManuelSHM(1,"GPU_FMCW/gpu_Manuel_FFT_Shared.txt");

    for(int i = 0; i < iTekrarSayisi; i++)
    {
        gpuManuelSHM.run_gpu_manuel_FFT_Shared_Mem(fullData);        
        gpuErrchk(cudaDeviceSynchronize());
    }

    FILE* file12 = fopen("GPU_FMCW/fftshared.txt", "w+");
    for (int i = 0; i < 2000; ++i) {
        fprintf(file12, "%-10d | %-12.4f | %-12.4f\n", i, gpuManuelSHM.getOutput()[i].real(), gpuManuelSHM.getOutput()[i].imag());
    }
    fclose(file12);

    memcpy(h_pinned_input, fullData.data(), TOTAL_ELEMENTS * sizeof(Complex));
    gpu_fmcw gpu2DFFT(2,"GPU_FMCW/2DFFT.txt");

    for(int i = 0; i < iTekrarSayisi; i++)
    {
        gpu2DFFT.run_gpu_2DFFT(h_pinned_input, h_pinned_outputGPU);
        gpuErrchk(cudaDeviceSynchronize());
    }


    FILE* file14 = fopen("GPU_FMCW/fft2D.txt", "w+");
    for (int i = 0; i < 2000; ++i) {
        fprintf(file14, "%-10d | %-12.4f | %-12.4f\n", i, gpu2DFFT.getOutput()[i].real(), gpu2DFFT.getOutput()[i].imag());
    }
    fclose(file14);


    cpu_fmcw cpuFMCWAVX("CPU_FMCW/cpu_AVX.txt");
    memcpy(h_pinned_input, fullData.data(), TOTAL_ELEMENTS * sizeof(Complex));
    cpuFMCWAVX.run_cpu_avx(h_pinned_input, h_pinned_outputCPU);
        
    FILE* file15 = fopen("CPU_FMCW/avxFFT.txt", "w+");
    for (int i = 0; i < 2000; ++i) {
        fprintf(file15, "%-10d | %-12.4f | %-12.4f\n", i, cpuFMCWAVX.getOutput()[i].real(), cpuFMCWAVX.getOutput()[i].imag());
    }
    fclose(file15);


    cpu_fmcw cpuFMCWManuel("CPU_FMCW/cpu_Recursive.txt");
    cpuFMCWManuel.run_cpu_basic(fullData);                
 

    FILE* file13 = fopen("CPU_FMCW/duzFFT.txt", "w+");
    for (int i = 0; i < 2000; ++i) {
        fprintf(file13, "%-10d | %-12.4f | %-12.4f\n", i, cpuFMCWManuel.getOutput()[i].real(), cpuFMCWManuel.getOutput()[i].imag());
    }
    fclose(file13);

    long long transferred_bytes = TOTAL_SIZE * sizeof(float) + TOTAL_ELEMENTS * sizeof(float);

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

    
    cudaFreeHost(h_pinned_input);
    cudaFreeHost(h_pinned_outputCPU);
    cudaFreeHost(h_pinned_outputGPU);

}