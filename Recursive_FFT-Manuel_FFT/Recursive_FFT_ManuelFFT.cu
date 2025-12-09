#include "Recursive_FFT_ManuelFFT.h"
#include "../kernels.h"
#include <cmath>

Recursive_FFT_Manuel_FFT::Recursive_FFT_Manuel_FFT()
{
    gpuErrchk(cudaMalloc(&d_data, TOTAL_SIZE * sizeof(cuComplex))); 
    gpuErrchk(cudaMalloc(&d_transposed, TOTAL_SIZE * sizeof(cuComplex)));
    cudaEventCreate(&start); 
    cudaEventCreate(&stop);
    log2_samples = (int)log2((double)NUM_SAMPLES);
    log2_chirps  = (int)log2((double)NUM_CHIRPS);
}

Recursive_FFT_Manuel_FFT::~Recursive_FFT_Manuel_FFT()
{
    cudaFree(d_data);
    cudaFree(d_transposed);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// DÜZELTİLMİŞ FONKSİYON: Artık geçici bir buffer (temp_ptr) kullanıyor
void Recursive_FFT_Manuel_FFT::execute_naive_fft(cuComplex* data_ptr, cuComplex* temp_ptr, int n, int batch_count, int log2_n)
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

void Recursive_FFT_Manuel_FFT::run_gpu_pipeline(const std::vector<Complex>& h_input, std::vector<Complex>& h_output)
{
    size_t sizeBytes = TOTAL_SIZE * sizeof(cuComplex);
    if(h_output.size() != TOTAL_SIZE) h_output.resize(TOTAL_SIZE);
    
    cudaEventRecord(start);

    // 1. Host -> Device
    gpuErrchk(cudaMemcpy(d_data, h_input.data(), sizeBytes, cudaMemcpyHostToDevice));

    // 2. Range FFT
    // Veri: d_data | Temp: d_transposed (Henüz boş, kullanabiliriz)
    execute_naive_fft(d_data, d_transposed, NUM_SAMPLES, NUM_CHIRPS, log2_samples);

    // 3. Transpose (d_data -> d_transposed)
    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks((NUM_SAMPLES + TILE_DIM - 1) / TILE_DIM, (NUM_CHIRPS + TILE_DIM - 1) / TILE_DIM);
    transpose_optimized_kernel<<<blocks, threads>>>(d_data, d_transposed, NUM_SAMPLES, NUM_CHIRPS);

    // 4. Doppler FFT
    // Veri: d_transposed | Temp: d_data (Range verisiyle işimiz bitti, çöplük yapabiliriz)
    execute_naive_fft(d_transposed, d_data, NUM_CHIRPS, NUM_SAMPLES, log2_chirps);

    // 5. Device -> Host (Sonuç d_transposed içinde)
    gpuErrchk(cudaMemcpy(h_output.data(), d_transposed, sizeBytes, cudaMemcpyDeviceToHost));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);
}

// ... CPU kodlarınız (run_cpu_pipeline vb.) aynı kalacak ...
// Buraya kopyalamadım, mevcut kodunuzdaki CPU kısmı doğrudur.

// Getter'lar
float Recursive_FFT_Manuel_FFT::getCpuTime() { return cpuTime; }
float Recursive_FFT_Manuel_FFT::getGPUTime() { return gpuTime; }
std::vector<Complex> Recursive_FFT_Manuel_FFT::getCpuResult() { return cpuResult; }
std::vector<Complex> Recursive_FFT_Manuel_FFT::getGpuResult() { return gpuResult; }
// CPU implementation (kopyala yapıştır yapabilirsiniz eskisiyle aynı)
void Recursive_FFT_Manuel_FFT::cpu_recursive_fft(std::vector<Complex>& a) {
    int n = a.size();
    if (n <= 1) return;
    std::vector<Complex> even(n / 2), odd(n / 2);
    for (int i = 0; 2 * i < n; i++) { even[i] = a[2 * i]; odd[i] = a[2 * i + 1]; }
    cpu_recursive_fft(even); cpu_recursive_fft(odd);
    for (int i = 0; i < n / 2; i++) {
        Complex t = std::polar(1.0f, -2.0f * PI * i / n) * odd[i];
        a[i] = even[i] + t; a[i + n / 2] = even[i] - t;
    }
}
void Recursive_FFT_Manuel_FFT::run_cpu_pipeline(const std::vector<Complex>& input, std::vector<Complex>& output) {
    std::vector<Complex> data = input; output.resize(TOTAL_SIZE);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_CHIRPS; ++i) {
        std::vector<Complex> row(NUM_SAMPLES);
        for(int j=0; j<NUM_SAMPLES; ++j) row[j] = data[i * NUM_SAMPLES + j];
        cpu_recursive_fft(row);
        for(int j=0; j<NUM_SAMPLES; ++j) data[i * NUM_SAMPLES + j] = row[j];
    }
    std::vector<Complex> transposed(TOTAL_SIZE);
    for (int i = 0; i < NUM_CHIRPS; ++i) {
        for (int j = 0; j < NUM_SAMPLES; ++j) transposed[j * NUM_CHIRPS + i] = data[i * NUM_SAMPLES + j];
    }
    for (int i = 0; i < NUM_SAMPLES; ++i) {
        std::vector<Complex> row(NUM_CHIRPS);
        for(int j=0; j<NUM_CHIRPS; ++j) row[j] = transposed[i * NUM_CHIRPS + j];
        cpu_recursive_fft(row);
        for(int j=0; j<NUM_CHIRPS; ++j) output[i * NUM_CHIRPS + j] = row[j];
    }
    auto end = std::chrono::high_resolution_clock::now();
    cpuTime = std::chrono::duration<float, std::milli>(end - start).count();
}