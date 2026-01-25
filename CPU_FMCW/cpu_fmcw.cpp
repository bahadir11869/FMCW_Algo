#include "cpu_fmcw.h"
#include <omp.h>



cpu_fmcw::cpu_fmcw(std::string strDosyaAdi)
{
    this->strDosyaAdi = strDosyaAdi;
    fcpuTime = 0.0;
    vfcpuTime = {};
    output.resize(TOTAL_SIZE);   
    handRange = NULL;
    handDoppler = NULL;

    DftiCreateDescriptor(&handRange, DFTI_SINGLE, DFTI_COMPLEX, 1, (MKL_LONG)NUM_SAMPLES);
    DftiSetValue(handRange, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)(NUM_CHIRPS * NUM_CHANNELS));
    DftiSetValue(handRange, DFTI_INPUT_DISTANCE, (MKL_LONG)NUM_SAMPLES);
    DftiSetValue(handRange, DFTI_OUTPUT_DISTANCE, (MKL_LONG)NUM_SAMPLES);
    DftiCommitDescriptor(handRange);

    // 3. Doppler FFT Planını bir kez oluştur
    DftiCreateDescriptor(&handDoppler, DFTI_SINGLE, DFTI_COMPLEX, 1, (MKL_LONG)NUM_CHIRPS);
    DftiSetValue(handDoppler, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)(NUM_SAMPLES * NUM_CHANNELS));
    DftiSetValue(handDoppler, DFTI_INPUT_DISTANCE, (MKL_LONG)NUM_CHIRPS);
    DftiSetValue(handDoppler, DFTI_OUTPUT_DISTANCE, (MKL_LONG)NUM_CHIRPS);
    DftiCommitDescriptor(handDoppler);

    Complex* all_transposed = nullptr;
}

cpu_fmcw::~cpu_fmcw()
{
    DftiFreeDescriptor(&handRange);
    DftiFreeDescriptor(&handDoppler);
    if(all_transposed) mkl_free(all_transposed);        
}

void cpu_fmcw::cpu_recursive_fft(std::vector<Complex>& a) 
{
    int n = a.size();
    if (n <= 1) return;
    std::vector<Complex> even(n / 2), odd(n / 2);

    for (int i = 0; 2 * i < n; i++) 
    {
        even[i] = a[2 * i];
        odd[i] = a[2 * i + 1]; 
    }
    cpu_recursive_fft(even); 
    cpu_recursive_fft(odd);
    for (int i = 0; i < n / 2; i++) 
    {
        Complex t = std::polar(1.0f, -2.0f * PI * i / n) * odd[i];
        a[i] = even[i] + t; a[i + n / 2] = even[i] - t;
    }
}

void cpu_fmcw::run_cpu_basic(const std::vector<Complex>& input)
{

    // Çıktıyı temizle ve boyutu ayarla (Tek bir kanal boyutunda)
    output.assign(TOTAL_SIZE, Complex(0.0f, 0.0f));
    
    auto start = std::chrono::high_resolution_clock::now();

    // 8 Kanalı tek tek dönüyoruz
    #pragma omp parallel for num_threads(NUM_CHANNELS)
    for (int ch = 0; ch < NUM_CHANNELS; ++ch) 
    {
        // 1. Mevcut kanalın verisini al (Offset hesaplama)
        // Python'da [TX][RX][Chirp][Sample] düzeninde kaydettiğimiz için:
        int channelOffset = ch * TOTAL_SIZE;
        std::vector<Complex> channelData(TOTAL_SIZE);
        for(int n = 0; n < TOTAL_SIZE; ++n) {
            channelData[n] = input[channelOffset + n];
        }

        // 2. Range FFT
        for (int i = 0; i < NUM_CHIRPS; ++i) {
            std::vector<Complex> row(NUM_SAMPLES);
            for(int j = 0; j < NUM_SAMPLES; ++j) 
                row[j] = channelData[i * NUM_SAMPLES + j];
            
            cpu_recursive_fft(row);
            
            for(int j = 0; j < NUM_SAMPLES; ++j) 
                channelData[i * NUM_SAMPLES + j] = row[j];
        }

        // 3. Transpose
        std::vector<Complex> transposed(TOTAL_SIZE);
        for (int i = 0; i < NUM_CHIRPS; ++i) {
            for (int j = 0; j < NUM_SAMPLES; ++j) 
                transposed[j * NUM_CHIRPS + i] = channelData[i * NUM_SAMPLES + j];
        }

        // 4. Doppler FFT ve Kanalları Toplama
        for (int i = 0; i < NUM_SAMPLES; ++i) {
            std::vector<Complex> row(NUM_CHIRPS);
            for(int j = 0; j < NUM_CHIRPS; ++j)
                row[j] = transposed[i * NUM_CHIRPS + j];
            
            cpu_recursive_fft(row);

            // İmajiner değerleri koruyarak ana output'a ekle (Coherent Sum)
            #pragma omp critical
            for(int j = 0; j < NUM_CHIRPS; ++j) {
                output[i * NUM_CHIRPS + j] += row[j];
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    fcpuTime = std::chrono::duration<float, std::milli>(end - start).count();
    vfcpuTime.push_back(fcpuTime);
}

void cpu_fmcw::run_cpu_openmp(const std::vector<Complex>& input) 
{
    std::vector<Complex> data = input;
    output.resize(TOTAL_SIZE);
    
    auto start = std::chrono::high_resolution_clock::now();

    // 1. RANGE FFT (OpenMP Parallel For)
    #pragma omp parallel for
    for (int i = 0; i < NUM_CHIRPS; ++i) 
    {
        std::vector<Complex> row(NUM_SAMPLES);
        for(int j=0; j<NUM_SAMPLES; ++j) 
            row[j] = data[i * NUM_SAMPLES + j];
        cpu_recursive_fft(row);
        for(int j=0; j<NUM_SAMPLES; ++j)
            data[i * NUM_SAMPLES + j] = row[j];
    }

    // 2. TRANSPOSE (OpenMP Collapse)
    std::vector<Complex> transposed(TOTAL_SIZE);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NUM_CHIRPS; ++i) 
    {
        for (int j = 0; j < NUM_SAMPLES; ++j) 
        {
            transposed[j * NUM_CHIRPS + i] = data[i * NUM_SAMPLES + j];
        }
    }

    // 3. DOPPLER FFT (OpenMP Parallel For)
    #pragma omp parallel for
    for (int i = 0; i < NUM_SAMPLES; ++i) 
    {
        std::vector<Complex> row(NUM_CHIRPS);
        for(int j=0; j<NUM_CHIRPS; ++j) 
            row[j] = transposed[i * NUM_CHIRPS + j];
        cpu_recursive_fft(row);
        for(int j=0; j<NUM_CHIRPS; ++j) 
            output[i * NUM_CHIRPS + j] = row[j];
    }

    auto end = std::chrono::high_resolution_clock::now();
    fcpuTime = std::chrono::duration<float, std::milli>(end - start).count();
    vfcpuTime.push_back(fcpuTime);
}

void cpu_fmcw::run_cpu_avx(Complex* input, Complex* ptroutput) 
{

    all_transposed = (Complex*)mkl_malloc(TOTAL_ELEMENTS * sizeof(Complex), 64);
    auto start = std::chrono::high_resolution_clock::now();
    // 1. In-place Range FFT (Input üzerinde direkt AVX kullanarak hesaplar)
    DftiComputeForward(handRange, (void*)input);
    // 2. Transpose (Burada OpenMP'yi sadece çok çekirdekli kazanç fazlaysa kullanın)
    #pragma omp parallel for collapse(3)
    for (int ch = 0; ch < NUM_CHANNELS; ++ch) {
        for (int i = 0; i < NUM_CHIRPS; ++i) {
            for (int j = 0; j < NUM_SAMPLES; ++j) {
                int in_idx = (ch * TOTAL_SIZE) + (i * NUM_SAMPLES + j);
                int out_idx = (ch * TOTAL_SIZE) + (j * NUM_CHIRPS + i);
                all_transposed[out_idx] = input[in_idx];
            }
        }
    }

    // 3. Doppler FFT
    DftiComputeForward(handDoppler, (void*)all_transposed);
    // 4. Coherent Summation
    std::memset(ptroutput, 0, TOTAL_SIZE * sizeof(Complex));
    #pragma omp parallel for
    for (int n = 0; n < TOTAL_SIZE; ++n) {
        Complex sum(0, 0);
        for (int ch = 0; ch < NUM_CHANNELS; ++ch) {
            sum += all_transposed[ch * TOTAL_SIZE + n];
        }
        ptroutput[n] = sum;
    }

    auto end = std::chrono::high_resolution_clock::now();
    fcpuTime = std::chrono::duration<float, std::milli>(end - start).count();
    vfcpuTime.push_back(fcpuTime);
    if(output.size() != TOTAL_SIZE) output.resize(TOTAL_SIZE);
    std::memcpy(output.data(), ptroutput, TOTAL_SIZE * sizeof(Complex));
}

float cpu_fmcw::getCpuTime()
{
    float fSum = 0.0f;
    for(auto i : vfcpuTime)
    {
        fSum += i;
    }
    return fSum/vfcpuTime.size();
}

std::string cpu_fmcw::getFileName()
{
    return strDosyaAdi;
}

std::vector<Complex> cpu_fmcw::getOutput()
{
    return output;
}