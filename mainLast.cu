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

    printf("Shared fft gecen sure %f \n", gpuManuelSHM.getGpuTime());

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

    printf("2DFFT gecen sure %f \n", gpu2DFFT.getGpuTime());

    FILE* file14 = fopen("GPU_FMCW/fft2D.txt", "w+");
    for (int i = 0; i < 2000; ++i) {
        fprintf(file14, "%-10d | %-12.4f | %-12.4f\n", i, gpu2DFFT.getOutput()[i].real(), gpu2DFFT.getOutput()[i].imag());
    }
    fclose(file14);


    cpu_fmcw cpuFMCWAVX("CPU_FMCW/cpu_AVX.txt");
    memcpy(h_pinned_input, fullData.data(), TOTAL_ELEMENTS * sizeof(Complex));
    for(int i = 0; i < iTekrarSayisi; i++)
    {
        cpuFMCWAVX.run_cpu_avx(h_pinned_input, h_pinned_outputCPU);
        
    }
    printf("AVX gecen sure %f \n", cpuFMCWAVX.getCpuTime());
    FILE* file15 = fopen("CPU_FMCW/avxFFT.txt", "w+");
    for (int i = 0; i < 2000; ++i) {
        fprintf(file15, "%-10d | %-12.4f | %-12.4f\n", i, cpuFMCWAVX.getOutput()[i].real(), cpuFMCWAVX.getOutput()[i].imag());
    }
    fclose(file15);


    cpu_fmcw cpuFMCWManuel("CPU_FMCW/cpu_Recursive.txt");
    for(int i = 0; i < iTekrarSayisi; i++)
    {
        cpuFMCWManuel.run_cpu_basic(fullData);                
    }
    printf("run_cpu_basic gecen sure %f \n", cpuFMCWManuel.getCpuTime());

    FILE* file13 = fopen("CPU_FMCW/duzFFT.txt", "w+");
    for (int i = 0; i < 2000; ++i) {
        fprintf(file13, "%-10d | %-12.4f | %-12.4f\n", i, cpuFMCWManuel.getOutput()[i].real(), cpuFMCWManuel.getOutput()[i].imag());
    }
    fclose(file13);

    /* SHM
    gpu_fmcw gpuManuelSHM(1,"GPU_FMCW/gpu_Manuel_FFT_Shared.txt");
    gpuManuelSHM.run_gpu_manuel_FFT_Shared_Mem(fullData);
    printf("gecen sure %f \n", gpuManuelSHM.getGpuTime());

    FILE* file12 = fopen("GPU_FMCW/fftshared.txt", "w+");
    for (int i = 0; i < 2000; ++i) {
        fprintf(file12, "%-10d | %-12.4f | %-12.4f\n", i, gpuManuelSHM.getOutput()[i].real(), gpuManuelSHM.getOutput()[i].imag());
    }
    fclose(file12);
    */
   /* 2DFFT


    memcpy(h_pinned_input, fullData.data(), TOTAL_ELEMENTS * sizeof(Complex));
    gpu_fmcw gpu2DFFT(2,"GPU_FMCW/2DFFT.txt");
    gpu2DFFT.run_gpu_2DFFT(h_pinned_input, h_pinned_outputGPU);
    printf("gecen sure %f \n", gpu2DFFT.getGpuTime());

    FILE* file14 = fopen("GPU_FMCW/fft2D.txt", "w+");
    for (int i = 0; i < 2000; ++i) {
        fprintf(file14, "%-10d | %-12.4f | %-12.4f\n", i, gpu2DFFT.getOutput()[i].real(), gpu2DFFT.getOutput()[i].imag());
    }
    fclose(file14);
    */

    /* OpenMP FFT
    cpu_fmcw cpuFMCWManuel("CPU_FMCW/cpu_Recursive.txt");
    cpuFMCWManuel.run_cpu_basic(fullData);
    printf("gecen sure %f \n", cpuFMCWManuel.getCpuTime());

    FILE* file13 = fopen("CPU_FMCW/duzFFT.txt", "w+");
    for (int i = 0; i < 2000; ++i) {
        fprintf(file13, "%-10d | %-12.4f | %-12.4f\n", i, cpuFMCWManuel.getOutput()[i].real(), cpuFMCWManuel.getOutput()[i].imag());
    }
    fclose(file13);
    */

    /*  18 ms openmp cpu
    
    cpu_fmcw cpuFMCWManuel("CPU_FMCW/cpu_Recursive.txt");
    cpuFMCWManuel.run_cpu_basic(fullData);
    printf("gecen sure %f \n", cpuFMCWManuel.getCpuTime());
    */

    /* 0.57 ms 2D fft
    Complex *h_pinned_input, *h_pinned_outputGPU, *h_pinned_outputCPU;
    gpuErrchk(cudaMallocHost((void**)&h_pinned_input, TOTAL_ELEMENTS * sizeof(Complex)));
    gpuErrchk(cudaMallocHost((void**)&h_pinned_outputCPU, TOTAL_SIZE * sizeof(Complex)));
    gpuErrchk(cudaMallocHost((void**)&h_pinned_outputGPU, TOTAL_SIZE * sizeof(Complex)));

    memcpy(h_pinned_input, fullData.data(), TOTAL_ELEMENTS * sizeof(Complex));
    gpu_fmcw gpu2DFFT(2,"GPU_FMCW/2DFFT.txt");
    gpu2DFFT.run_gpu_2DFFT(h_pinned_input, h_pinned_outputGPU);
    printf("gecen sure %f \n", gpu2DFFT.getGpuTime());
    */

    /*
    long long total_bytes = TOTAL_SIZE * sizeof(float);
    long long transferred_bytes = 2 * total_bytes;

    std::vector<Complex> inputData(TOTAL_SIZE);
    std::vector<Complex> outputData(TOTAL_SIZE);

    Complex *h_pinned_input, *h_pinned_outputGPU, *h_pinned_outputCPU;
    gpuErrchk(cudaMallocHost((void**)&h_pinned_input, TOTAL_SIZE * sizeof(Complex)));
    gpuErrchk(cudaMallocHost((void**)&h_pinned_outputCPU, TOTAL_SIZE * sizeof(Complex)));
    gpuErrchk(cudaMallocHost((void**)&h_pinned_outputGPU, TOTAL_SIZE * sizeof(Complex)));

    veriUret(inputData, s1);   

    cpu_fmcw cpuFMCWManuel("CPU_FMCW/cpu_Recursive.txt");
    cpu_fmcw cpuFMCWOpenMP("CPU_FMCW/cpu_Recursive_OPENMP.txt");
    cpu_fmcw cpuFMCWAVX("CPU_FMCW/cpu_AVX.txt");

    gpu_fmcw gpuManuelNoSHM(1, "GPU_FMCW/gpu_Manuel_FFT_SharedYok.txt");
    gpu_fmcw gpuManuelSHM(1,"GPU_FMCW/gpu_Manuel_FFT_Shared.txt");
    gpu_fmcw gpuManuelSHMStream(1,"GPU_FMCW/gpu_Manuel_FFT_Shared_Stream.txt");
    gpu_fmcw gpu1DCufftManuelTranspose(1, "GPU_FMCW/gpu_1DFFT_ManuelTranspose.txt");
    gpu_fmcw gpu2DFFT(2,"GPU_FMCW/2DFFT.txt");

    int iTekrarSayisi = 5;
    
    for(int i = 0; i < iTekrarSayisi; i++)
    {
        gpuManuelNoSHM.run_gpu_manuel_FFT_Shared_Yok(inputData);
        gpuErrchk(cudaDeviceSynchronize());
    }
    
    for(int i = 0; i < iTekrarSayisi; i++)
    {
        gpuManuelSHM.run_gpu_manuel_FFT_Shared_Mem(inputData);
        gpuErrchk(cudaDeviceSynchronize());
    }

    for(int i = 0; i < iTekrarSayisi; i++)
    {
        gpuManuelSHMStream.run_gpu_streams(inputData.data(), outputData.data());
        gpuErrchk(cudaDeviceSynchronize());
    }
    
    for(int i = 0; i < iTekrarSayisi; i++)
    {
        gpu1DCufftManuelTranspose.run_gpu_manuel_transpose(inputData);
        gpuErrchk(cudaDeviceSynchronize());
    }

    memcpy(h_pinned_input, inputData.data(), TOTAL_SIZE* sizeof(Complex));
    for(int i = 0; i < iTekrarSayisi; i++)
    {
        gpu2DFFT.run_gpu_2DFFT(h_pinned_input, h_pinned_outputGPU);
        gpuErrchk(cudaDeviceSynchronize());
    }
    
    for(int i = 0; i < iTekrarSayisi; i++)
    {
        cpuFMCWManuel.run_cpu_basic(inputData); 
    }
    

    for(int i = 0; i < iTekrarSayisi; i++)
    {
        cpuFMCWOpenMP.run_cpu_openmp(inputData);
    }
    
    for(int i = 0; i < iTekrarSayisi; i++)
    {
        cpuFMCWAVX.run_cpu_avx(h_pinned_input, h_pinned_outputCPU);
    }


    save_rdm_data("GPU_FMCW/2DFFT_GPU.csv", gpu2DFFT.getOutput(), false);
    save_rdm_data("CPU_FMCW/AVXFFT_CPU.csv", cpuFMCWAVX.getOutput(), true);  
 
    dosyayaYaz(gpuManuelNoSHM.getDosyaAdi().c_str(), gpuManuelNoSHM.getOutput(), s1, 100.0);
    dosyayaYaz(gpuManuelSHM.getDosyaAdi().c_str(), gpuManuelSHM.getOutput(), s1, 100.0);
    dosyayaYaz(gpuManuelSHMStream.getDosyaAdi().c_str(), gpuManuelSHMStream.getOutput(), s1, 100.0);
    dosyayaYaz(gpu1DCufftManuelTranspose.getDosyaAdi().c_str(), gpu1DCufftManuelTranspose.getOutput(), s1, 100.0);
    dosyayaYaz(gpu2DFFT.getDosyaAdi().c_str(), gpu2DFFT.getOutput(), s1, 100.0, true);

    dosyayaYaz(cpuFMCWManuel.getFileName().c_str(), cpuFMCWManuel.getOutput(), s1, 100.0);
    dosyayaYaz(cpuFMCWOpenMP.getFileName().c_str(), cpuFMCWOpenMP.getOutput(), s1, 100.0);
    dosyayaYaz(cpuFMCWAVX.getFileName().c_str(), cpuFMCWAVX.getOutput(), s1, 100.0);
      
    printf("\n\n\t\t\t\t\t\t\t\t\t --------SONUCLAR GPU --------- \n\n");
    printf("GPU Manuel FFT Shared yok: Compute time: %f ms total time: %f ms bandWithGPUManuel : %f GB/s RTX 3060 Max BandWith: ~360 GB/s \n", gpuManuelNoSHM.getGpuComputeTime(), gpuManuelNoSHM.getGpuTime(), float(transferred_bytes * 1e-9)/(gpuManuelNoSHM.getGpuComputeTime()/1000.0));    
    printf("GPU Manuel FFT Shared Mem: Compute time: %f ms  total time: %f ms bandWithGPUManuelSharedMem : %f GB/s RTX 3060 Max BandWith: ~360 GB/s \n", gpuManuelSHM.getGpuComputeTime(),  gpuManuelSHM.getGpuTime(), (transferred_bytes * 1e-9)/(gpuManuelSHM.getGpuComputeTime()/1000.0));    
    printf("GPU Manuel FFT Shared Stream: Compute time: %f ms  total time: %f ms bandWithGPUManuelSharedMem : %f GB/s RTX 3060 Max BandWith: ~360 GB/s \n", gpuManuelSHMStream.getGpuComputeTime(),  gpuManuelSHMStream.getGpuTime(), (transferred_bytes * 1e-9)/(gpuManuelSHMStream.getGpuComputeTime()/1000.0));
    printf("GPU 1DFFT Manuel Transpose : Compute time: %f ms total time: %f ms  bandWithGPUManuelTranspose : %f GB/s RTX 3060 Max BandWith: ~360 GB/s\n", gpu1DCufftManuelTranspose.getGpuComputeTime(), gpu1DCufftManuelTranspose.getGpuTime(), (transferred_bytes * 1e-9)/(gpu1DCufftManuelTranspose.getGpuComputeTime()/1000.0));    
    printf("2D_FFT Compute time: %f ms total time: %f ms bandWithGPU2D : %f GB/s RTX 3060 Max BandWith: ~360 GB/s\n", gpu2DFFT.getGpuComputeTime(),gpu2DFFT.getGpuTime(), (transferred_bytes * 1e-9)/(gpu2DFFT.getGpuComputeTime()/1000.0));
    
    printf("\n\n\t\t\t\t\t\t\t\t\t --------SONUCLAR CPU --------- \n\n");
    printf("CPU Recursive FFT time: %f CPU Recurisive FFT OpenMP time: %f CPU AVX Total time: %f\n", 
    cpuFMCWManuel.getCpuTime(), cpuFMCWOpenMP.getCpuTime(), cpuFMCWAVX.getCpuTime());
    
    printf("\n\n\t\t\t\t\t\t\t\t\t --------SONUCLAR CPU AVX vs GPU  --------- \n\n");

    printf("GPUManuelFFTGlobal vs AVX RMSE \n");        
    calculate_RMSE_Vectors(cpuFMCWAVX.getOutput(), gpuManuelNoSHM.getOutput(), NUM_SAMPLES, NUM_CHIRPS, false);  
    
    printf("GPUManuelFFTSharedMemStream vs AVX RMSE \n");        
    calculate_RMSE_Vectors(cpuFMCWAVX.getOutput(), gpuManuelSHMStream.getOutput(), NUM_SAMPLES, NUM_CHIRPS, false);  

    printf("GPUManuelFFTSharedMem vs AVX RMSE \n");
    calculate_RMSE_Vectors(cpuFMCWAVX.getOutput(), gpuManuelSHM.getOutput(), NUM_SAMPLES, NUM_CHIRPS, false);  

    printf("GPUManuelFFTStream vs AVX RMSE \n");
    calculate_RMSE_Vectors(cpuFMCWAVX.getOutput(), gpuManuelSHMStream.getOutput(), NUM_SAMPLES, NUM_CHIRPS, false);  
    
    printf("GpuManuelTranspose vs AVX RMSE \n");
    calculate_RMSE_Vectors(cpuFMCWAVX.getOutput(), gpu1DCufftManuelTranspose.getOutput(), NUM_SAMPLES, NUM_CHIRPS, false);  

    printf("Gpu2D vs AVX RMSE \n");
    calculate_RMSE_Vectors(cpuFMCWAVX.getOutput(), gpu2DFFT.getOutput(), NUM_SAMPLES, NUM_CHIRPS, true);  

    
    cudaFreeHost(h_pinned_input);
    cudaFreeHost(h_pinned_outputCPU);
    cudaFreeHost(h_pinned_outputGPU);

    */
}