#include "gpu_fmcw.h"
#include "../KERNEL_FMCW/kernels.h"


gpu_fmcw::gpu_fmcw(int fftType, std::string strDosyaAdi)
{
    this->strDosyaAdi = strDosyaAdi;
    vfgpuTime = {};
    vfgpuComputeTime = {};
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    this->fftType = fftType;
    gpuErrchk(cudaMalloc(&d_data, TOTAL_SIZE * sizeof(cuComplex))); 
    log2_samples = (int)log2((double)NUM_SAMPLES);
    log2_chirps  = (int)log2((double)NUM_CHIRPS);
    output.resize(TOTAL_SIZE);   
    for (int i = 0; i < 4; ++i) 
    {
        cudaStreamCreate(&streams[i]);
    }

    if(fftType == 1)
    {
        cufftPlan1d(&planRange, NUM_SAMPLES, CUFFT_C2C, NUM_CHIRPS);
        cufftPlan1d(&planDoppler, NUM_CHIRPS, CUFFT_C2C, NUM_SAMPLES);
        gpuErrchk(cudaMalloc(&d_transposed, TOTAL_SIZE * sizeof(cuComplex)));    
    }
    else
    {
        if (cufftPlan2d(&plan, NUM_CHIRPS, NUM_SAMPLES, CUFFT_C2C) != CUFFT_SUCCESS) 
        {
            std::cerr << "CUFFT Plan Hatasi!" << std::endl;
            exit(1);    
        }
    }    
}


gpu_fmcw::~gpu_fmcw()
{
    cudaFree(d_data);    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);

    for (int i = 0; i < 4; ++i) 
    {
        cudaStreamDestroy(streams[i]);
    }

    if(fftType == 1)
    {
        cufftDestroy(planRange); 
        cufftDestroy(planDoppler);
        cudaFree(d_transposed);
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
void gpu_fmcw::run_gpu_manuel_FFT_Shared_Yok(std::vector<Complex>& input)
{
    size_t sizeBytes = TOTAL_SIZE * sizeof(cuComplex);
    if(input.size() != TOTAL_SIZE) 
        output.resize(TOTAL_SIZE);
    
    cudaEventRecord(start);

    // 1. Host -> Device
    gpuErrchk(cudaMemcpy(d_data, input.data(), sizeBytes, cudaMemcpyHostToDevice));

    // 2. Range FFT
    // Veri: d_data | Temp: d_transposed (Henüz boş, kullanabiliriz)
    cudaEventRecord(start2);
    execute_naive_fft(d_data, d_transposed, NUM_SAMPLES, NUM_CHIRPS, log2_samples);

    // 3. Transpose (d_data -> d_transposed)
    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks((NUM_SAMPLES + TILE_DIM - 1) / TILE_DIM, (NUM_CHIRPS + TILE_DIM - 1) / TILE_DIM);
    transpose_optimized_kernel<<<blocks, threads>>>(d_data, d_transposed, NUM_SAMPLES, NUM_CHIRPS);

    // 4. Doppler FFT
    // Veri: d_transposed | Temp: d_data (Range verisiyle işimiz bitti, çöplük yapabiliriz)
    execute_naive_fft(d_transposed, d_data, NUM_CHIRPS, NUM_SAMPLES, log2_chirps);
    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);
    // 5. Device -> Host (Sonuç d_transposed içinde)
    gpuErrchk(cudaMemcpy(output.data(), d_transposed, sizeBytes, cudaMemcpyDeviceToHost));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);


    cudaEventElapsedTime(&fgpuTime, start, stop);
    cudaEventElapsedTime(&fgpuComputeTime, start2, stop2);
    vfgpuTime.push_back(fgpuTime);
    vfgpuComputeTime.push_back(fgpuComputeTime);
}

void gpu_fmcw::run_gpu_manuel_FFT_Shared_Mem(std::vector<Complex>& input)
{
    size_t sizeBytes = TOTAL_SIZE * sizeof(cuComplex);
    if(input.size() != TOTAL_SIZE) output.resize(TOTAL_SIZE);
    
    cudaEventRecord(start);

    // 1. Host -> Device
    gpuErrchk(cudaMemcpy(d_data, input.data(), sizeBytes, cudaMemcpyHostToDevice));
    
    cudaEventRecord(start2);

    // --- RANGE FFT (Shared Memory Optimized) ---
    // Her blok 1 Chirp işleyecek. (NUM_CHIRPS tane blok)
    // Her blokta N/2 tane thread olacak. (1024 / 2 = 512 thread)
    // Shared Memory Boyutu: N * sizeof(cuComplex) -> 1024 * 8 = 8KB (Yeterli)
    
    int threadsPerBlock = NUM_SAMPLES / 2;
    int numBlocks = NUM_CHIRPS;
    int sharedMemSize = NUM_SAMPLES * sizeof(cuComplex);

    k_fft_shared<<<numBlocks, threadsPerBlock, sharedMemSize>>>(d_data, NUM_SAMPLES, log2_samples);
    gpuErrchk(cudaGetLastError()); // Hata kontrolü

    // 3. Transpose (Mevcut kerneliniz çok iyi, aynen kalsın)
    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks((NUM_SAMPLES + TILE_DIM - 1) / TILE_DIM, (NUM_CHIRPS + TILE_DIM - 1) / TILE_DIM);
    transpose_optimized_kernel<<<blocks, threads>>>(d_data, d_transposed, NUM_SAMPLES, NUM_CHIRPS);

    // 4. Doppler FFT (Shared Memory Optimized)
    // Artık veri d_transposed içinde [Sample][Chirp] şeklinde.
    // FFT boyutu = NUM_CHIRPS (2048).
    // Thread sayısı = 1024.
    // Blok sayısı = NUM_SAMPLES (1024 tane satır işleyeceğiz).
    
    threadsPerBlock = NUM_CHIRPS / 2; 
    numBlocks = NUM_SAMPLES;
    sharedMemSize = NUM_CHIRPS * sizeof(cuComplex); // 16KB (Bu da sığar)

    // Doppler FFT'yi transpoze edilmiş veri üzerinde çalıştırıyoruz
    // Çıktıyı tekrar d_data'ya yazabiliriz (yer kazanmak için)
    // NOT: Doppler FFT sonucu d_transposed üzerinde oluşacak (In-Place olduğu için)
    // Ama kernel d_data üzerinde çalışsın istiyorsak parametreleri değiştirebiliriz.
    // Basitlik için d_transposed üzerinde yapıp sonucu ordan alalım.
    
    k_fft_shared<<<numBlocks, threadsPerBlock, sharedMemSize>>>(d_transposed, NUM_CHIRPS, log2_chirps);
    gpuErrchk(cudaGetLastError());

    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);

    // 5. Device -> Host
    // Sonuç d_transposed içinde kaldı.
    gpuErrchk(cudaMemcpy(output.data(), d_transposed, sizeBytes, cudaMemcpyDeviceToHost));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&fgpuTime, start, stop);
    cudaEventElapsedTime(&fgpuComputeTime, start2, stop2);
    vfgpuTime.push_back(fgpuTime);
    vfgpuComputeTime.push_back(fgpuComputeTime);
}

void gpu_fmcw::run_gpu_streams(Complex* h_input, Complex* h_output)
{
    // Veriyi 4 parçaya bölüyoruz
    int nStreams = 4;
    int chirpsPerStream = NUM_CHIRPS / nStreams; // 2048 / 4 = 512 Chirp
    int samplesPerStream = chirpsPerStream * NUM_SAMPLES; // Veri boyutu
    size_t streamBytes = samplesPerStream * sizeof(cuComplex);

    cudaEventRecord(start); // Süreyi başlat

    // --- AŞAMA 1: Range FFT + Veri Yükleme (PIPELINE) ---
    // Döngü dönerken: Bir stream yükleme yaparken diğeri hesaplama yapacak.
    
    cudaEventRecord(start2); // Compute süresi (Transfer dahil pipeline)

    for (int i = 0; i < nStreams; ++i) 
    {
        int offset = i * samplesPerStream; // Verinin başlangıç noktası

        // 1. ASENKRON Kopyalama (Host -> Device)
        // CPU beklemez, hemen bir sonraki satıra geçer.
        cudaMemcpyAsync(&d_data[offset], &h_input[offset], streamBytes, cudaMemcpyHostToDevice, streams[i]);

        // 2. Range FFT (Sadece o parçayı işle)
        // Kernelin o parçada çalışması için pointer'ı kaydırıyoruz (d_data + offset)
        int threadsPerBlock = NUM_SAMPLES / 2;
        int numBlocks = chirpsPerStream; // Sadece 512 blok
        int sharedMemSize = NUM_SAMPLES * sizeof(cuComplex);

        k_fft_shared<<<numBlocks, threadsPerBlock, sharedMemSize, streams[i]>>>(d_data + offset, NUM_SAMPLES, log2_samples);
    }

    // DİKKAT: Transpose işlemi tüm verinin bitmesini beklemek ZORUNDADIR.
    // Çünkü satır verisi sütuna dönüşecek, veriler karışacak.
    cudaDeviceSynchronize(); 

    // --- AŞAMA 2: Transpose (Mevcut Kernel - Aynen kalıyor) ---
    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks((NUM_SAMPLES + TILE_DIM - 1) / TILE_DIM, (NUM_CHIRPS + TILE_DIM - 1) / TILE_DIM);
    transpose_optimized_kernel<<<blocks, threads>>>(d_data, d_transposed, NUM_SAMPLES, NUM_CHIRPS);
    
    // --- AŞAMA 3: Doppler FFT ---
    // Transpose sonrası veri d_transposed içinde. Bunu çalıştırıyoruz.
    int threadsPerBlock = NUM_CHIRPS / 2; 
    int numBlocks = NUM_SAMPLES;
    int sharedMemSize = NUM_CHIRPS * sizeof(cuComplex);
    
    // Doppler FFT (Sonuç d_transposed üzerinde kalıyor)
    k_fft_shared<<<numBlocks, threadsPerBlock, sharedMemSize>>>(d_transposed, NUM_CHIRPS, log2_chirps);
    
    cudaEventRecord(stop2); // Compute Bitiş

    // --- AŞAMA 4: Sonuç Kopyalama (Device -> Host) ---
    // Bunu da istersen stream ile bölebilirsin ama tek seferde alalım şimdilik.
    cudaMemcpy(h_output, d_transposed, TOTAL_SIZE * sizeof(cuComplex), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop); // Toplam Bitiş
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&fgpuTime, start, stop);
    cudaEventElapsedTime(&fgpuComputeTime, start2, stop2);
    vfgpuTime.push_back(fgpuTime);
    vfgpuComputeTime.push_back(fgpuComputeTime);
    memcpy(output.data(), h_output, TOTAL_SIZE * sizeof(cuComplex));
}

void gpu_fmcw::run_gpu_manuel_transpose(std::vector<Complex>& input)
{
    size_t sizeBytes = TOTAL_SIZE * sizeof(cuComplex);
    output.resize(TOTAL_SIZE);
    
    cudaEventRecord(start);
    gpuErrchk(cudaMemcpy(d_data, input.data(), sizeBytes, cudaMemcpyHostToDevice));
    cudaEventRecord(start2);
    cufftExecC2C(planRange, d_data, d_data, CUFFT_FORWARD);
    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks((NUM_SAMPLES + TILE_DIM - 1) / TILE_DIM, (NUM_CHIRPS + TILE_DIM - 1) / TILE_DIM);
    transpose_optimized_kernel<<<blocks, threads>>>(d_data, d_transposed, NUM_SAMPLES, NUM_CHIRPS);
    cufftExecC2C(planDoppler, d_transposed, d_transposed, CUFFT_FORWARD);
    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);
    gpuErrchk(cudaMemcpy(output.data(), d_transposed, sizeBytes, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&fgpuTime, start, stop);
    cudaEventElapsedTime(&fgpuComputeTime, start2, stop2);
    vfgpuTime.push_back(fgpuTime);
    vfgpuComputeTime.push_back(fgpuComputeTime);
}

void gpu_fmcw::run_gpu_2DFFT(Complex* input, Complex* ptroutput)
{
    size_t sizeBytes = TOTAL_SIZE * sizeof(cuComplex);

    // --- KRITIK BOLGE BASLANGIC ---
    cudaEventRecord(start);

    // 1. Veri Transferi (Host -> Device)
    // Pinned memory olduğu için çok daha hızlıdır.
    gpuErrchk(cudaMemcpy(d_data, input, sizeBytes, cudaMemcpyHostToDevice));
    cudaEventRecord(start2);
    // 2. 2D FFT (Hem Range hem Doppler işlemini ve Transpose mantığını içerir)
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);
    // 3. Veri Transferi (Device -> Host)
    gpuErrchk(cudaMemcpy(ptroutput, d_data, sizeBytes, cudaMemcpyDeviceToHost));

    cudaEventRecord(stop);
    // --- KRITIK BOLGE BITIS ---

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&fgpuTime, start, stop);
    cudaEventElapsedTime(&fgpuComputeTime, start2, stop2);
    vfgpuTime.push_back(fgpuTime);
    vfgpuComputeTime.push_back(fgpuComputeTime);
    memcpy(output.data(), ptroutput, sizeBytes);
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
float gpu_fmcw::getGpuComputeTime()
{
    float fSum = 0.0f;
    for(auto i: vfgpuComputeTime)
    {
        fSum += i;
    }
    return fSum / vfgpuComputeTime.size();
}

std::string gpu_fmcw::getDosyaAdi()
{
    return strDosyaAdi;
}


std::vector<Complex> gpu_fmcw::getOutput()
{
    return output;    
}
