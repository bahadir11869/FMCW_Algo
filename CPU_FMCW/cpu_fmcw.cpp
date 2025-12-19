#include "cpu_fmcw.h"
#include <omp.h>
#include <mkl.h>


cpu_fmcw::cpu_fmcw(std::string strDosyaAdi)
{
    this->strDosyaAdi = strDosyaAdi;
    fcpuTime = 0.0;
    vfcpuTime = {};
    output.resize(TOTAL_SIZE);   
}

cpu_fmcw::~cpu_fmcw()
{
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
    std::vector<Complex> data = input; output.resize(TOTAL_SIZE);
    auto start = std::chrono::high_resolution_clock::now();
    // 1. Range FFT
    for (int i = 0; i < NUM_CHIRPS; ++i) 
    {
        std::vector<Complex> row(NUM_SAMPLES);
        for(int j=0; j<NUM_SAMPLES; ++j) 
            row[j] = data[i * NUM_SAMPLES + j];
        cpu_recursive_fft(row);
        for(int j=0; j<NUM_SAMPLES; ++j) 
            data[i * NUM_SAMPLES + j] = row[j];
    }
    // 2. Transpose
    std::vector<Complex> transposed(TOTAL_SIZE);
    for (int i = 0; i < NUM_CHIRPS; ++i)
    {
        for (int j = 0; j < NUM_SAMPLES; ++j) 
            transposed[j * NUM_CHIRPS + i] = data[i * NUM_SAMPLES + j];
    }
    // 3. Doppler FFT
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
   std::memcpy(ptroutput, input, TOTAL_SIZE * sizeof(Complex));
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
    status = DftiComputeForward(handRange, (void*)ptroutput);

    // --- 3. TRANSPOSE (MKL OMATCOPY veya OPENMP) ---
    // MKL'in matris çevirme fonksiyonu da var ama OpenMP daha anlaşılır.
    // Out-of-place transpose: data -> output
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NUM_CHIRPS; ++i) {
        for (int j = 0; j < NUM_SAMPLES; ++j) {
            temp_buffer[j * NUM_CHIRPS + i] = ptroutput[i * NUM_SAMPLES + j];
        }
    }

    // --- 4. DOPPLER FFT İŞLEMİ ---
    // Transpoze edilmiş 'output' verisi üzerinde çalışıyoruz.
    status = DftiComputeForward(handDoppler, (void*)temp_buffer);
    auto end = std::chrono::high_resolution_clock::now();
    // --- TEMİZLİK ---
    DftiFreeDescriptor(&handRange);
    DftiFreeDescriptor(&handDoppler);

    
    std::memcpy(ptroutput, temp_buffer, TOTAL_SIZE * sizeof(Complex));
    delete[] temp_buffer;
    fcpuTime = std::chrono::duration<float, std::milli>(end - start).count();
    vfcpuTime.push_back(fcpuTime);
    memcpy(output.data(), ptroutput, TOTAL_SIZE * sizeof(Complex));
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