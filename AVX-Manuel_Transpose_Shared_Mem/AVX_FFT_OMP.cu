#include "AVX_FFT_OMP.h"
#include "../kernels.h"
#include <omp.h>
#include <mkl.h> // Intel MKL başlık dosyası

AVX_FFT_OpenMP::AVX_FFT_OpenMP()
{
    cufftPlan1d(&planRange, NUM_SAMPLES, CUFFT_C2C, NUM_CHIRPS);
    cufftPlan1d(&planDoppler, NUM_CHIRPS, CUFFT_C2C, NUM_SAMPLES);
    gpuErrchk(cudaMalloc(&d_data, TOTAL_SIZE * sizeof(cuComplex))); gpuErrchk(cudaMalloc(&d_transposed, TOTAL_SIZE * sizeof(cuComplex)));
}

AVX_FFT_OpenMP::~AVX_FFT_OpenMP()
{
    cufftDestroy(planRange); cufftDestroy(planDoppler); cudaFree(d_data); cudaFree(d_transposed);
}


void AVX_FFT_OpenMP::run_gpu_pipeline(const std::vector<Complex> h_input, std::vector<Complex>& h_output)
{
    size_t sizeBytes = TOTAL_SIZE * sizeof(cuComplex);
    h_output.resize(TOTAL_SIZE);
    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
    gpuErrchk(cudaMemcpy(d_data, h_input.data(), sizeBytes, cudaMemcpyHostToDevice));

    cudaEventRecord(start);
    cufftExecC2C(planRange, d_data, d_data, CUFFT_FORWARD);
    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks((NUM_SAMPLES + TILE_DIM - 1) / TILE_DIM, (NUM_CHIRPS + TILE_DIM - 1) / TILE_DIM);
    transpose_optimized_kernel<<<blocks, threads>>>(d_data, d_transposed, NUM_SAMPLES, NUM_CHIRPS);
    cufftExecC2C(planDoppler, d_transposed, d_transposed, CUFFT_FORWARD);

    gpuErrchk(cudaMemcpy(h_output.data(), d_transposed, sizeBytes, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);
   
}

void AVX_FFT_OpenMP::run_cpu_pipeline(const std::vector<Complex> input, std::vector<Complex>& output)
{
    std::vector<Complex> data = input;
    output.resize(TOTAL_SIZE); // Doppler sonucu için yer ayır
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
    status = DftiComputeForward(handRange, (void*)data.data());

    // --- 3. TRANSPOSE (MKL OMATCOPY veya OPENMP) ---
    // MKL'in matris çevirme fonksiyonu da var ama OpenMP daha anlaşılır.
    // Out-of-place transpose: data -> output
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NUM_CHIRPS; ++i) {
        for (int j = 0; j < NUM_SAMPLES; ++j) {
            output[j * NUM_CHIRPS + i] = data[i * NUM_SAMPLES + j];
        }
    }

    // --- 4. DOPPLER FFT İŞLEMİ ---
    // Transpoze edilmiş 'output' verisi üzerinde çalışıyoruz.
    status = DftiComputeForward(handDoppler, (void*)output.data());

    // --- TEMİZLİK ---
    DftiFreeDescriptor(&handRange);
    DftiFreeDescriptor(&handDoppler);

    auto end = std::chrono::high_resolution_clock::now();
    cpuTime = std::chrono::duration<float, std::milli>(end - start).count();
}

float AVX_FFT_OpenMP::getCpuTime()
{
    return cpuTime;
}

float AVX_FFT_OpenMP::getGPUTime()
{
    return gpuTime;
}

std::vector<Complex> AVX_FFT_OpenMP::getCpuResult()
{
    return cpuResult;
}

std::vector<Complex> AVX_FFT_OpenMP::getGpuResult()
{
    return gpuResult;
}
