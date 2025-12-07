#include "Recursive_FFT.h"
#include "../kernels.h"


Recursive_FFT::Recursive_FFT()
{
    cufftPlan1d(&planRange, NUM_SAMPLES, CUFFT_C2C, NUM_CHIRPS);
    cufftPlan1d(&planDoppler, NUM_CHIRPS, CUFFT_C2C, NUM_SAMPLES);
    gpuErrchk(cudaMalloc(&d_data, TOTAL_SIZE * sizeof(cuComplex))); 
    gpuErrchk(cudaMalloc(&d_transposed, TOTAL_SIZE * sizeof(cuComplex)));
    cudaEventCreate(&start); 
    cudaEventCreate(&stop);
}

Recursive_FFT::~Recursive_FFT()
{
    cufftDestroy(planRange); 
    cufftDestroy(planDoppler); 
    cudaFree(d_data);
    cudaFree(d_transposed);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


void Recursive_FFT::run_gpu_pipeline(const std::vector<Complex>& h_input, std::vector<Complex>& h_output)
{
    size_t sizeBytes = TOTAL_SIZE * sizeof(cuComplex);
    h_output.resize(TOTAL_SIZE);
    cudaEventRecord(start);

    gpuErrchk(cudaMemcpy(d_data, h_input.data(), sizeBytes, cudaMemcpyHostToDevice));
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

void Recursive_FFT::cpu_recursive_fft(std::vector<Complex>& a) {
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

void Recursive_FFT::run_cpu_pipeline(const std::vector<Complex>& input, std::vector<Complex>& output)
{
   std::vector<Complex> data = input; output.resize(TOTAL_SIZE);
    auto start = std::chrono::high_resolution_clock::now();
    // 1. Range FFT
    for (int i = 0; i < NUM_CHIRPS; ++i) {
        std::vector<Complex> row(NUM_SAMPLES);
        for(int j=0; j<NUM_SAMPLES; ++j) row[j] = 
            data[i * NUM_SAMPLES + j];
        cpu_recursive_fft(row);
        for(int j=0; j<NUM_SAMPLES; ++j) 
            data[i * NUM_SAMPLES + j] = row[j];
    }
    // 2. Transpose
    std::vector<Complex> transposed(TOTAL_SIZE);
    for (int i = 0; i < NUM_CHIRPS; ++i) {
        for (int j = 0; j < NUM_SAMPLES; ++j) transposed[j * NUM_CHIRPS + i] = data[i * NUM_SAMPLES + j];
    }
    // 3. Doppler FFT
    for (int i = 0; i < NUM_SAMPLES; ++i) {
        std::vector<Complex> row(NUM_CHIRPS);
        for(int j=0; j<NUM_CHIRPS; ++j) row[j] = transposed[i * NUM_CHIRPS + j];
        cpu_recursive_fft(row);
        for(int j=0; j<NUM_CHIRPS; ++j) output[i * NUM_CHIRPS + j] = row[j];
    }
    auto end = std::chrono::high_resolution_clock::now();
    cpuTime = std::chrono::duration<float, std::milli>(end - start).count();
}

float Recursive_FFT::getCpuTime()
{
    return cpuTime;
}

float Recursive_FFT::getGPUTime()
{
    return gpuTime;
}

std::vector<Complex> Recursive_FFT::getCpuResult()
{
    return cpuResult;
}

std::vector<Complex> Recursive_FFT::getGpuResult()
{
    return gpuResult;
}
