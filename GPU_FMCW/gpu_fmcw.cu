#include "gpu_fmcw.h"
#include "../KERNEL_FMCW/kernels.h"


gpu_fmcw::gpu_fmcw(int fftType)
{
    vfgpuTime = {};
    vfgpuComputeTime = {};
    cudaEventCreate(&start_total);
    cudaEventCreate(&stop_total);
    cudaEventCreate(&start_fmcw_compute);
    cudaEventCreate(&stop_fmcw_compute);
    cudaEventCreate(&start_cfar_compute);
    cudaEventCreate(&stop_cfar_compute);

    this->fftType = fftType;
    gpuErrchk(cudaMalloc(&f_data, TOTAL_SIZE * sizeof(float))); 
    f_data_host = new float[TOTAL_SIZE];
    //gpuErrchk(cudaMalloc(&f_data_host, TOTAL_SIZE * sizeof(float))); 
    gpuErrchk(cudaMalloc(&d_data, TOTAL_SIZE * sizeof(cuComplex))); 
    gpuErrchk(cudaMalloc(&d_data_all, TOTAL_ELEMENTS * sizeof(cuComplex))); 
    gpuErrchk(cudaMalloc(&d_transposed, TOTAL_SIZE * sizeof(cuComplex)));    
    log2_samples = (int)log2((double)NUM_SAMPLES);
    log2_chirps  = (int)log2((double)NUM_CHIRPS);
    output.resize(TOTAL_SIZE);   
    gpuErrchk(cudaMalloc(&d_transposed_all, TOTAL_ELEMENTS * sizeof(cuComplex)));    
    for (int i = 0; i < 4; ++i) 
    {
        cudaStreamCreate(&streams[i]);
    }

    if(fftType == 1)
    {
        cufftPlan1d(&planRange, NUM_SAMPLES, CUFFT_C2C, NUM_CHIRPS);
        cufftPlan1d(&planDoppler, NUM_CHIRPS, CUFFT_C2C, NUM_SAMPLES);
       
    }
    else
    {
        int n[2] = {NUM_CHIRPS, NUM_SAMPLES};
        cufftPlanMany(&plan, 2, n, 
                NULL, 1, TOTAL_SIZE, // Input layout
                NULL, 1, TOTAL_SIZE, // Output layout
                CUFFT_C2C, NUM_CHANNELS); // 8 Adet (Batch)

    } 

    gpuErrchk(cudaMalloc(&fpCfarData, TOTAL_SIZE * sizeof(float))); 
    gpuErrchk(cudaMalloc(&bpCfarData, TOTAL_SIZE * sizeof(bool))); 


    bpCFAR = new bool[TOTAL_SIZE];
    cfarProcessor.init(cfarParam);
}


gpu_fmcw::~gpu_fmcw()
{
    cudaFree(d_data);
    cudaFree(d_data_all);
    cudaFree(f_data);
    cudaFree(d_transposed);
    cudaFree(d_transposed_all);
    cudaFree(fpCfarData);
    cudaFree(bpCfarData);
    delete [] f_data_host;
    delete [] bpCFAR;
    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
    cudaEventDestroy(start_fmcw_compute);
    cudaEventDestroy(stop_fmcw_compute);
    cudaEventDestroy(start_cfar_compute);
    cudaEventDestroy(stop_cfar_compute);

    for (int i = 0; i < 4; ++i)
    {
        cudaStreamDestroy(streams[i]);
    }

    if(fftType == 1)
    {
        cufftDestroy(planRange);
        cufftDestroy(planDoppler);
    }
    else
    {
        cufftDestroy(plan);
    }
}

void gpu_fmcw::execute_naive_fft(cuComplex* data_ptr, cuComplex* temp_ptr, int n, int batch_count, int log2_n)
{
    // 1. Bit-Reversal (Sıralama)
    // Kaynaktan (data_ptr) oku -> Hedefe (temp_ptr) sıralı yaz
    dim3 threads(256);
    dim3 blocks((n + 255) / 256, batch_count);
    
    k_bit_reversal<<<blocks, threads>>>(data_ptr, temp_ptr, n, log2_n, batch_count);
    
    // Sıralanmış veriyi tekrar ana pointer'a geri al
    gpuErrchk(cudaMemcpy(data_ptr, temp_ptr, n * batch_count * sizeof(cuComplex), cudaMemcpyDeviceToDevice));

    // 2. Butterfly Stages (Yerinde - In Place)
    dim3 blocks_bf((n / 2 + 255) / 256, batch_count);

    for (int stage_width = 1; stage_width < n; stage_width *= 2) {
        k_butterfly_stage<<<blocks_bf, threads>>>(data_ptr, n, stage_width, batch_count);
    }
}

void gpu_fmcw::run_gpu_manuel_FFT_Shared_Mem(std::vector<Complex>& input, float* fOutput)
{
    size_t sizeBytes = TOTAL_ELEMENTS * sizeof(cuComplex);
    cudaEventRecord(start_total);

    // 1. Host -> Device
    gpuErrchk(cudaMemcpy(d_data_all, input.data(), sizeBytes, cudaMemcpyHostToDevice));
    cudaEventRecord(start_fmcw_compute);

    int threadsPerBlock = NUM_SAMPLES / 2;
    int numBlocks = NUM_CHIRPS * NUM_CHANNELS;
    int sharedMemSize = NUM_SAMPLES * sizeof(cuComplex);

    k_fft_shared<<<numBlocks, threadsPerBlock, sharedMemSize>>>(d_data_all, NUM_SAMPLES, log2_samples);

    dim3 threads(TILE_DIM, NUM_CHANNELS);
    // Grid'in Z boyutuna kanal sayısını ekliyoruz
    dim3 blocks((NUM_SAMPLES + TILE_DIM - 1) / TILE_DIM, 
                (NUM_CHIRPS + TILE_DIM - 1) / TILE_DIM, 
                NUM_CHANNELS);

    // Not: Transpose kernel'ınızın içinde d_data_all + blockIdx.z * TOTAL_SIZE ofsetini kullanmalısınız
    transpose_multi_channel_kernel<<<blocks, threads>>>(d_data_all, d_transposed_all, NUM_SAMPLES, NUM_CHIRPS);


    // 4. Doppler FFT (Shared Memory Optimized)
    // Artık veri d_transposed içinde [Sample][Chirp] şeklinde.
    // FFT boyutu = NUM_CHIRPS (2048).
    // Thread sayısı = 1024.
    // Blok sayısı = NUM_SAMPLES (1024 tane satır işleyeceğiz).
    
    threadsPerBlock = NUM_CHIRPS / 2; 
    numBlocks = NUM_SAMPLES * NUM_CHANNELS;
    sharedMemSize = NUM_CHIRPS * sizeof(cuComplex); // 16KB (Bu da sığar)

    // Doppler FFT'yi transpoze edilmiş veri üzerinde çalıştırıyoruz
    // Çıktıyı tekrar d_data'ya yazabiliriz (yer kazanmak için)
    // NOT: Doppler FFT sonucu d_transposed üzerinde oluşacak (In-Place olduğu için)
    // Ama kernel d_data üzerinde çalışsın istiyorsak parametreleri değiştirebiliriz.
    // Basitlik için d_transposed üzerinde yapıp sonucu ordan alalım.
    
    k_fft_shared<<<numBlocks, threadsPerBlock, sharedMemSize>>>(d_transposed_all, NUM_CHIRPS, log2_chirps);

    // Kanal toplama + FFTShift + yakın mesafe supresyonu (i < 5)
    dim3 threadsSum(16, 16);
    dim3 blocksSum(
        (NUM_SAMPLES + threadsSum.x - 1) / threadsSum.x,
        (NUM_CHIRPS  + threadsSum.y - 1) / threadsSum.y
    );
    sumChannelsShiftKernel<<<blocksSum, threadsSum>>>(
        d_transposed_all,
        fpCfarData,
        NUM_CHANNELS,
        NUM_SAMPLES,
        NUM_CHIRPS
    );
  
    cudaEventRecord(stop_fmcw_compute);
    cudaEventSynchronize(stop_fmcw_compute);

    cudaEventRecord(start_cfar_compute);    
    cfarProcessor.process(cfarParam, fpCfarData, bpCfarData);
    cudaEventRecord(stop_cfar_compute);
    cudaEventSynchronize(stop_cfar_compute);

    cudaMemcpy(bpCFAR, bpCfarData, TOTAL_SIZE * sizeof(bool), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop_total);
    cudaEventSynchronize(stop_total);

    cudaEventElapsedTime(&fgpuTime, start_total, stop_total);
    cudaEventElapsedTime(&fgpuComputeTime, start_fmcw_compute, stop_fmcw_compute);
    cudaEventElapsedTime(&cfarComputeTime, start_cfar_compute, stop_cfar_compute);

    vfgpuTime.push_back(fgpuTime);
    vfgpuComputeTime.push_back(fgpuComputeTime);
    vfCfargpuTime.push_back(cfarComputeTime);
    
    gpuErrchk(cudaMemcpy(fOutput, fpCfarData, TOTAL_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    memcpy(f_data_host, fOutput,  TOTAL_SIZE * sizeof(float));
}

void gpu_fmcw::run_gpu_2DFFT(Complex* input, float* ptroutput)
{
    size_t sizeBytes = TOTAL_ELEMENTS * sizeof(cuComplex);

    // --- KRITIK BOLGE BASLANGIC ---
    cudaEventRecord(start_total);

    // 1. Veri Transferi (Host -> Device)
    // Pinned memory olduğu için çok daha hızlıdır.
    gpuErrchk(cudaMemcpy(d_data_all, input, sizeBytes, cudaMemcpyHostToDevice));
    
    cudaEventRecord(start_fmcw_compute);
    // 2. 2D FFT (Hem Range hem Doppler işlemini ve Transpose mantığını içerir)
    cufftExecC2C(plan, d_data_all, d_data_all, CUFFT_FORWARD);


    dim3 threads(TILE_DIM, NUM_CHANNELS);
    // Grid'in Z boyutuna kanal sayısını ekliyoruz
    dim3 blocks((NUM_SAMPLES + TILE_DIM - 1) / TILE_DIM, 
                (NUM_CHIRPS + TILE_DIM - 1) / TILE_DIM, 
                NUM_CHANNELS);

    // Not: Transpose kernel'ınızın içinde d_data_all + blockIdx.z * TOTAL_SIZE ofsetini kullanmalısınız
    transpose_multi_channel_kernel<<<blocks, threads>>>(d_data_all, d_transposed_all, NUM_SAMPLES, NUM_CHIRPS);

    // Kanal toplama + FFTShift + yakın mesafe supresyonu (i < 5)
    dim3 threadsSum2D(16, 16);
    dim3 blocksSum2D(
        (NUM_SAMPLES + threadsSum2D.x - 1) / threadsSum2D.x,
        (NUM_CHIRPS  + threadsSum2D.y - 1) / threadsSum2D.y
    );
    sumChannelsShiftKernel<<<blocksSum2D, threadsSum2D>>>(
        d_transposed_all,
        fpCfarData,
        NUM_CHANNELS,
        NUM_SAMPLES,
        NUM_CHIRPS
    );
    cudaEventRecord(stop_fmcw_compute);
    cudaEventSynchronize(stop_fmcw_compute);

    cudaEventRecord(start_cfar_compute);
    cfarProcessor.process(cfarParam, fpCfarData, bpCfarData);
    cudaEventRecord(stop_cfar_compute);

    cudaMemcpy(bpCFAR, bpCfarData, TOTAL_SIZE * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop_total);
    cudaEventSynchronize(stop_total);

    gpuErrchk(cudaMemcpy(ptroutput, fpCfarData, TOTAL_SIZE * sizeof(float), cudaMemcpyDeviceToHost));


    cudaEventElapsedTime(&fgpuComputeTime, start_fmcw_compute, stop_fmcw_compute);
    vfgpuComputeTime.push_back(fgpuComputeTime);

    // --- KRITIK BOLGE BITIS ---
    cudaEventElapsedTime(&fgpuTime, start_total, stop_total);
    vfgpuTime.push_back(fgpuTime);

    cudaEventElapsedTime(&cfarComputeTime, start_cfar_compute, stop_cfar_compute);
    vfCfargpuTime.push_back(cfarComputeTime);

    memcpy(f_data_host, ptroutput,  TOTAL_SIZE * sizeof(float));
    //memcpy(output.data(), ptroutput, TOTAL_SIZE * sizeof(Complex));
}


float gpu_fmcw::getGpuTime()
{
    float fSum = 0.0f;
    for(auto i: vfgpuTime)
    {
        fSum += i;
    }
    return fSum / vfgpuTime.size();
}

float gpu_fmcw::getGpuTimeTotal()
{
    float fSum = 0.0f;
    for(auto i: vfgpuTime)
    {
        fSum += i;
    }
    return fSum;
}

float gpu_fmcw::getGpuComputeTime()
{
    float fSum = 0.0f;
    for(auto i: vfgpuComputeTime)
    {
        fSum += i;
    }
    return fSum / vfgpuComputeTime.size();
}
float gpu_fmcw::getCfarGpuComputeTime()
{
    float fSum = 0.0f;
    for(auto i: vfCfargpuTime)
    {
        fSum += i;
    }
    return fSum / vfCfargpuTime.size();
}


float* gpu_fmcw::getOutput()
{
    return f_data_host;    
}
