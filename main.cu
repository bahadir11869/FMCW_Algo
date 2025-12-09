// nvcc main.cu --options-file compile.txt;./myapp.exe
#include "Recursive_FFT-Manuel_FFT/Recursive_FFT_ManuelFFT.h"
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

    std::vector<Complex> inputData(TOTAL_SIZE);
    std::vector<Complex> outputDataCpu(TOTAL_SIZE);
    std::vector<Complex> outputDataGpu(TOTAL_SIZE);
    veriUret(inputData, s1);        
    int iTekrarSayisi = 5;
    float fCpuTime = 0.0;
    float fGpuTime = 0.0;
    const char* dosyaAdi = "";

    {      // Recursive_FFT_Manuel_FFT
        Recursive_FFT_Manuel_FFT Recursive_FFT_Manuel_FFT_GPU;

        for(int i = 0; i <iTekrarSayisi; i++)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(15));
            Recursive_FFT_Manuel_FFT_GPU.run_cpu_pipeline(inputData, outputDataCpu);
            fCpuTime += Recursive_FFT_Manuel_FFT_GPU.getCpuTime();
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        Recursive_FFT_Manuel_FFT_GPU.run_gpu_pipeline(inputData, outputDataGpu);
        gpuErrchk(cudaDeviceSynchronize());

        for(int i = 0; i <iTekrarSayisi; i++)
        {
            Recursive_FFT_Manuel_FFT_GPU.run_gpu_pipeline(inputData, outputDataGpu);
            gpuErrchk(cudaDeviceSynchronize());
            fGpuTime += Recursive_FFT_Manuel_FFT_GPU.getGPUTime();
            //printf("Recursive_FFT-Manuel_Transpose_Shared_Mem [%d] %f \n", i, RecursiveFFT_ManuelTranspose.getGPUTime());
        }
        dosyaAdi = "Recursive_FFT-Manuel_FFT/cpu.txt";
        fs::remove(dosyaAdi);
        dosyayaYaz(dosyaAdi, fCpuTime / (float)iTekrarSayisi,  fGpuTime / (float)iTekrarSayisi, outputDataCpu, s1, 100.0);
        dosyaAdi = "Recursive_FFT-Manuel_FFT/gpu.txt";
        fs::remove(dosyaAdi);
        dosyayaYaz(dosyaAdi, fCpuTime / (float)iTekrarSayisi,  fGpuTime / (float)iTekrarSayisi, outputDataGpu, s1, 100.0);

        printf("\n\nRecursive_FFT_Manuel_FFT ortalama cpu suresi : %f ms ortalama gpu suresi %f ms\n", fCpuTime / (float)iTekrarSayisi,  fGpuTime / (float)iTekrarSayisi);
    }

    fCpuTime = 0.0;
    fGpuTime = 0.0;
    gpuErrchk(cudaDeviceSynchronize());
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    {      // Recursive_FFT-Manuel_Transpose_Shared_Mem
        Recursive_FFT RecursiveFFT_ManuelTranspose;

        for(int i = 0; i <iTekrarSayisi; i++)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(15));
            RecursiveFFT_ManuelTranspose.run_cpu_pipeline(inputData, outputDataCpu);
            fCpuTime += RecursiveFFT_ManuelTranspose.getCpuTime();
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        RecursiveFFT_ManuelTranspose.run_gpu_pipeline(inputData, outputDataGpu);
        gpuErrchk(cudaDeviceSynchronize());

        for(int i = 0; i <iTekrarSayisi; i++)
        {
            
            RecursiveFFT_ManuelTranspose.run_gpu_pipeline(inputData, outputDataGpu);
            gpuErrchk(cudaDeviceSynchronize());
            fGpuTime += RecursiveFFT_ManuelTranspose.getGPUTime();
            //printf("Recursive_FFT-Manuel_Transpose_Shared_Mem [%d] %f \n", i, RecursiveFFT_ManuelTranspose.getGPUTime());
        }
        dosyaAdi = "Recursive_FFT-Manuel_Transpose_Shared_Mem/cpu.txt";
        fs::remove(dosyaAdi);
        dosyayaYaz(dosyaAdi, fCpuTime / (float)iTekrarSayisi,  fGpuTime / (float)iTekrarSayisi, outputDataCpu, s1, 100.0);
        dosyaAdi = "Recursive_FFT-Manuel_Transpose_Shared_Mem/gpu.txt";
        fs::remove(dosyaAdi);
        dosyayaYaz(dosyaAdi, fCpuTime / (float)iTekrarSayisi,  fGpuTime / (float)iTekrarSayisi, outputDataGpu, s1, 100.0);

        printf("\n\nRecursive_FFT-Manuel_Transpose_Shared_Mem ortalama cpu suresi : %f ms ortalama gpu suresi %f ms\n", fCpuTime / (float)iTekrarSayisi,  fGpuTime / (float)iTekrarSayisi);
    }

    fCpuTime = 0.0;
    fGpuTime = 0.0;
    gpuErrchk(cudaDeviceSynchronize());
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    {    // OpenMP-Manuel_Transpose_Shared_Mem
        Recursive_FFT_OpenMP RecursiveFFT_ManuelTranspose_OpenMP;
        for(int i = 0; i <iTekrarSayisi; i++)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(15));
            RecursiveFFT_ManuelTranspose_OpenMP.run_cpu_pipeline(inputData, outputDataCpu);
            fCpuTime += RecursiveFFT_ManuelTranspose_OpenMP.getCpuTime();
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        RecursiveFFT_ManuelTranspose_OpenMP.run_gpu_pipeline(inputData, outputDataGpu);
        gpuErrchk(cudaDeviceSynchronize());

        for(int i = 0; i <iTekrarSayisi; i++)
        {
            //std::this_thread::sleep_for(std::chrono::milliseconds(15));
            RecursiveFFT_ManuelTranspose_OpenMP.run_gpu_pipeline(inputData, outputDataGpu);
            gpuErrchk(cudaDeviceSynchronize());
            fGpuTime += RecursiveFFT_ManuelTranspose_OpenMP.getGPUTime();
            //printf("OpenMP-Manuel_Transpose_Shared_Mem [%d] %f \n", i, RecursiveFFT_ManuelTranspose_OpenMP.getGPUTime());
        }

        dosyaAdi = "OpenMP-Manuel_Transpose_Shared_Mem/cpu.txt";
        fs::remove(dosyaAdi);
        dosyayaYaz(dosyaAdi, fCpuTime / (float)iTekrarSayisi,  fGpuTime / (float)iTekrarSayisi, outputDataCpu, s1, 100.0);
        dosyaAdi =  "OpenMP-Manuel_Transpose_Shared_Mem/gpu.txt";
        fs::remove(dosyaAdi);
        dosyayaYaz(dosyaAdi, fCpuTime / (float)iTekrarSayisi,  fGpuTime / (float)iTekrarSayisi, outputDataGpu, s1, 100.0);

        printf("\n\nOpenMP-Manuel_Transpose_Shared_Mem ortalama cpu suresi : %f ms ortalama gpu suresi %f ms\n", fCpuTime / (float)iTekrarSayisi,  fGpuTime / (float)iTekrarSayisi);
    }
    fCpuTime = 0.0;
    fGpuTime = 0.0;
    gpuErrchk(cudaDeviceSynchronize());
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    { // AVX_FFT_OpenMP_ManuelTranspose
        AVX_FFT_OpenMP AVX_FFT_OpenMP_ManuelTranspose;
         for(int i = 0; i <iTekrarSayisi; i++)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(15));
            AVX_FFT_OpenMP_ManuelTranspose.run_cpu_pipeline(inputData, outputDataCpu);
            fCpuTime += AVX_FFT_OpenMP_ManuelTranspose.getCpuTime();
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        AVX_FFT_OpenMP_ManuelTranspose.run_gpu_pipeline(inputData, outputDataGpu);
        gpuErrchk(cudaDeviceSynchronize());

        for(int i = 0; i <iTekrarSayisi; i++)
        {
            
            //std::this_thread::sleep_for(std::chrono::milliseconds(15));
            AVX_FFT_OpenMP_ManuelTranspose.run_gpu_pipeline(inputData, outputDataGpu);
            gpuErrchk(cudaDeviceSynchronize());
            fGpuTime += AVX_FFT_OpenMP_ManuelTranspose.getGPUTime();
        }
        
        dosyaAdi = "AVX-Manuel_Transpose_Shared_Mem/cpu.txt";
        fs::remove(dosyaAdi);
        dosyayaYaz(dosyaAdi, fCpuTime / (float)iTekrarSayisi,  fGpuTime / (float)iTekrarSayisi, outputDataCpu, s1, 100.0);
        dosyaAdi = "AVX-Manuel_Transpose_Shared_Mem/gpu.txt";
        fs::remove(dosyaAdi);
        dosyayaYaz(dosyaAdi, fCpuTime / (float)iTekrarSayisi,  fGpuTime / (float)iTekrarSayisi, outputDataGpu, s1, 100.0);

        printf("\n\nAVX_FFT_OpenMP_ManuelTranspose ortalama cpu suresi : %f ms ortalama gpu suresi %f ms\n", fCpuTime / (float)iTekrarSayisi,  fGpuTime / (float)iTekrarSayisi);
    }

    fCpuTime = 0.0;
    fGpuTime = 0.0;
    gpuErrchk(cudaDeviceSynchronize());
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    {     // AVX-2DFFT
        AVX_FFT_OpenMP_2DFFT AVX_FFT_OpenMP_OtoTranspose;
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
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        AVX_FFT_OpenMP_OtoTranspose.run_gpu_pipeline(h_pinned_input, h_pinned_outputGPU);
        gpuErrchk(cudaDeviceSynchronize());

        for(int i = 0; i <iTekrarSayisi; i++)
        {
            //std::this_thread::sleep_for(std::chrono::milliseconds(15));
            AVX_FFT_OpenMP_OtoTranspose.run_gpu_pipeline(h_pinned_input, h_pinned_outputGPU);
            gpuErrchk(cudaDeviceSynchronize());
            fGpuTime += AVX_FFT_OpenMP_OtoTranspose.getGPUTime();
        }
        
        memcpy(outputDataGpu.data(), h_pinned_outputGPU, sizeof(Complex) * TOTAL_SIZE);
        memcpy(outputDataCpu.data(), h_pinned_outputCPU, sizeof(Complex) * TOTAL_SIZE);
        dosyaAdi = "AVX-2DFFT/cpu.txt";
        fs::remove(dosyaAdi);
        dosyayaYaz(dosyaAdi, fCpuTime / (float)iTekrarSayisi,  fGpuTime / (float)iTekrarSayisi, outputDataCpu, s1, 100.0, false);
        dosyaAdi = "AVX-2DFFT/gpu.txt";
        fs::remove(dosyaAdi);
        dosyayaYaz(dosyaAdi, fCpuTime / (float)iTekrarSayisi,  fGpuTime / (float)iTekrarSayisi, outputDataGpu, s1, 100.0, true);

        printf("\n\nAVX-2DFFT ortalama cpu suresi : %f ms ortalama gpu suresi %f ms\n", fCpuTime / (float)iTekrarSayisi,  fGpuTime / (float)iTekrarSayisi);
        cudaFreeHost(h_pinned_input);
        cudaFreeHost(h_pinned_outputCPU);
        cudaFreeHost(h_pinned_outputGPU);
   
    }


    return 0;
}