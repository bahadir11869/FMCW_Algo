#include "AVX_FFT_OpenMP_2DFFT.h"
#include <omp.h>
#include <mkl.h> // Intel MKL başlık dosyası

AVX_FFT_OpenMP_2DFFT::AVX_FFT_OpenMP_2DFFT()
{
    printf("AVX_FFT_OpenMP_2DFFT \n");
    gpuErrchk(cudaMalloc(&d_data, TOTAL_SIZE * sizeof(cuComplex)));
    if (cufftPlan2d(&plan, NUM_CHIRPS, NUM_SAMPLES, CUFFT_C2C) != CUFFT_SUCCESS) 
    {
        std::cerr << "CUFFT Plan Hatasi!" << std::endl;
        exit(1);    
    }
}

AVX_FFT_OpenMP_2DFFT::~AVX_FFT_OpenMP_2DFFT()
{
    cudaFree(d_data);
    cufftDestroy(plan);
}


void AVX_FFT_OpenMP_2DFFT::run_gpu_pipeline(Complex* h_input, Complex* h_output)
{
    size_t sizeBytes = TOTAL_SIZE * sizeof(cuComplex);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // --- KRITIK BOLGE BASLANGIC ---
    cudaEventRecord(start);

    // 1. Veri Transferi (Host -> Device)
    // Pinned memory olduğu için çok daha hızlıdır.
    gpuErrchk(cudaMemcpy(d_data, h_input, sizeBytes, cudaMemcpyHostToDevice));

    // 2. 2D FFT (Hem Range hem Doppler işlemini ve Transpose mantığını içerir)
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);

    // 3. Veri Transferi (Device -> Host)
    gpuErrchk(cudaMemcpy(h_output, d_data, sizeBytes, cudaMemcpyDeviceToHost));

    cudaEventRecord(stop);
    // --- KRITIK BOLGE BITIS ---

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void AVX_FFT_OpenMP_2DFFT::run_cpu_pipeline(Complex* input, Complex* output)
{
    std::memcpy(output, input, TOTAL_SIZE * sizeof(Complex));
    Complex* temp_buffer = new Complex[TOTAL_SIZE];
    auto start = std::chrono::high_resolution_clock::now();
    // --- MKL CONFIGURATION ---
    DFTI_DESCRIPTOR_HANDLE handRange = NULL;
    DFTI_DESCRIPTOR_HANDLE handDoppler = NULL;
    MKL_LONG status;

    // A. RANGE FFT PLAN (BATCH MODE)
    // 1D FFT, Uzunluk: NUM_SAMPLES, Tip: Complex-to-Complex
    status = DftiCreateDescriptor(&handRange, DFTI_SINGLE, DFTI_COMPLEX, 1, (MKL_LONG)NUM_SAMPLES);
    
    // Batch Ayarı: Aynı anda NUM_CHIRPS (256) tane FFT yap
    status = DftiSetValue(handRange, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)NUM_CHIRPS);
    // Input Distance: Bir sonraki FFT verisi ne kadar uzakta? (NUM_SAMPLES kadar)
    status = DftiSetValue(handRange, DFTI_INPUT_DISTANCE, (MKL_LONG)NUM_SAMPLES);
    // Output Distance: Sonucu nereye yazayım? (Yerine yazıyoruz - InPlace)
    status = DftiSetValue(handRange, DFTI_OUTPUT_DISTANCE, (MKL_LONG)NUM_SAMPLES);
    
    status = DftiCommitDescriptor(handRange); // Planı kilitle

    // B. DOPPLER FFT PLAN (BATCH MODE)
    // 1D FFT, Uzunluk: NUM_CHIRPS
    status = DftiCreateDescriptor(&handDoppler, DFTI_SINGLE, DFTI_COMPLEX, 1, (MKL_LONG)NUM_CHIRPS);
    
    // Batch Ayarı: Aynı anda NUM_SAMPLES (512) tane FFT yap
    status = DftiSetValue(handDoppler, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)NUM_SAMPLES);
    status = DftiSetValue(handDoppler, DFTI_INPUT_DISTANCE, (MKL_LONG)NUM_CHIRPS);
    status = DftiSetValue(handDoppler, DFTI_OUTPUT_DISTANCE, (MKL_LONG)NUM_CHIRPS);
    
    status = DftiCommitDescriptor(handDoppler);

    // --- 2. RANGE FFT İŞLEMİ ---
    // DftiComputeForward: Tek satırda 256 tane FFT'yi AVX-512 ile hesaplar.
    // std::complex* -> void* dönüşümü yapıyoruz.
    status = DftiComputeForward(handRange, (void*)output);

    // --- 3. TRANSPOSE (MKL OMATCOPY veya OPENMP) ---
    // MKL'in matris çevirme fonksiyonu da var ama OpenMP daha anlaşılır.
    // Out-of-place transpose: data -> output
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NUM_CHIRPS; ++i) {
        for (int j = 0; j < NUM_SAMPLES; ++j) {
            temp_buffer[j * NUM_CHIRPS + i] = output[i * NUM_SAMPLES + j];
        }
    }

    // --- 4. DOPPLER FFT İŞLEMİ ---
    // Transpoze edilmiş 'output' verisi üzerinde çalışıyoruz.
    status = DftiComputeForward(handDoppler, (void*)temp_buffer);
    auto end = std::chrono::high_resolution_clock::now();
    // --- TEMİZLİK ---
    DftiFreeDescriptor(&handRange);
    DftiFreeDescriptor(&handDoppler);

    
    std::memcpy(output, temp_buffer, TOTAL_SIZE * sizeof(Complex));
    delete[] temp_buffer;
    cpuTime = std::chrono::duration<float, std::milli>(end - start).count();
}

float AVX_FFT_OpenMP_2DFFT::getCpuTime()
{
    return cpuTime;
}

float AVX_FFT_OpenMP_2DFFT::getGPUTime()
{
    return gpuTime;
}

std::vector<Complex> AVX_FFT_OpenMP_2DFFT::getCpuResult()
{
    return cpuResult;
}

std::vector<Complex> AVX_FFT_OpenMP_2DFFT::getGpuResult()
{
    return gpuResult;
}
