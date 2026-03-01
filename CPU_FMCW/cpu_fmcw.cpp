#include "cpu_fmcw.h"
#include <omp.h>


cpu_fmcw::cpu_fmcw() : cpu_sat(p), avxcfar(p)
{
    fmcwCpuTime = 0.0f;
    cfarCpuTime =0.0f;
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

    all_transposed = nullptr;

    sumVector = new float[TOTAL_SIZE];
    memset(sumVector, 0, TOTAL_SIZE * sizeof(float));    
    cfarData.truth = new bool[TOTAL_SIZE];
    memset(cfarData.truth,false, TOTAL_SIZE * sizeof(bool));    
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
    memset(sumVector, 0, TOTAL_SIZE * sizeof(float));
    auto start = std::chrono::high_resolution_clock::now();

    // 8 Kanalı tek tek dönüyoruz
    #pragma omp parallel for num_threads(NUM_CHANNELS)
    for (int ch = 0; ch < NUM_CHANNELS; ++ch) 
    {
        int channelOffset = ch * TOTAL_SIZE;
        std::vector<Complex> channelData(TOTAL_SIZE);
        for(int n = 0; n < TOTAL_SIZE; ++n) {
            channelData[n] = input[channelOffset + n];
        }

        // 2. Range FFT (pencere veri kaynağında uygulandı)
        for (int i = 0; i < NUM_CHIRPS; ++i) {
            std::vector<Complex> row(NUM_SAMPLES);
            for(int j = 0; j < NUM_SAMPLES; ++j) {
                row[j] = channelData[i * NUM_SAMPLES + j];
            }
            
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
        // Her thread kendi lokal buffer'ına yazar → critical section yok
        std::vector<float> localSum(TOTAL_SIZE, 0.0f);

        for (int i = 0; i < NUM_SAMPLES; ++i) {
            std::vector<Complex> row(NUM_CHIRPS);

            // Veriyi kopyala
            for(int j = 0; j < NUM_CHIRPS; ++j) {
                row[j] = transposed[i * NUM_CHIRPS + j];
            }
            
            // MTI: Doppler FFT öncesi chirp ortalamasını çıkar (statik clutter baskılama)
            Complex mean_val(0.0f, 0.0f);
            for(int j = 0; j < NUM_CHIRPS; ++j) mean_val += row[j];
            mean_val /= (float)NUM_CHIRPS;
            for(int j = 0; j < NUM_CHIRPS; ++j) row[j] -= mean_val;

            cpu_recursive_fft(row);

            // Güç hesabı (Non-Coherent Integration)
            // i < 5: yakın mesafe (DC sızıntısı) → atla, localSum zaten 0
            if(i >= 5)
            {
                for(int j = 0; j < NUM_CHIRPS; ++j) 
                {
                    // FFTSHIFT: 0-Doppler'i merkeze kaydır
                    int shifted_j = (j + NUM_CHIRPS / 2) % NUM_CHIRPS; 
                    localSum[i * NUM_CHIRPS + shifted_j] += std::norm(row[j]);
                }
            }
        }

        // Thread-lokal sonuçları global sumVector'e tek seferde topla
        #pragma omp critical
        for(int k = 0; k < TOTAL_SIZE; ++k)
            sumVector[k] += localSum[k];
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto cfar = std::chrono::high_resolution_clock::now(); 
    cfarData.power = std::vector<float>(sumVector, sumVector + TOTAL_SIZE);
    cpu_sat.process(cfarData);
    auto cfar_end = std::chrono::high_resolution_clock::now();
    applyPeakRelativeFilter(cfarData.truth, cfarData.power.data(), NUM_SAMPLES, NUM_CHIRPS);

    fmcwCpuTime = std::chrono::duration<float, std::milli>(end - start).count();
    cfarCpuTime =  std::chrono::duration<float, std::milli>(cfar_end - cfar).count();
    vfcpuTime.push_back(fmcwCpuTime);
    vCfarCpuTime.push_back(cfarCpuTime);
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

    // 3. MTI: Doppler FFT öncesi her (kanal, range_bin) için chirp ortalamasını çıkar
    #pragma omp parallel for collapse(2)
    for (int ch = 0; ch < NUM_CHANNELS; ++ch) {
        for (int i = 0; i < NUM_SAMPLES; ++i) {
            int base = ch * TOTAL_SIZE + i * NUM_CHIRPS;
            Complex mean_val(0.0f, 0.0f);
            for (int j = 0; j < NUM_CHIRPS; ++j)
                mean_val += all_transposed[base + j];
            mean_val /= (float)NUM_CHIRPS;
            for (int j = 0; j < NUM_CHIRPS; ++j)
                all_transposed[base + j] -= mean_val;
        }
    }

    // 4. Doppler FFT
    DftiComputeForward(handDoppler, (void*)all_transposed);
    // 4. Coherent Summation + FFTShift + yakın mesafe supresyonu (i < 5)
    std::memset(ptroutput, 0, TOTAL_SIZE * sizeof(Complex));
    memset(sumVector, 0, TOTAL_SIZE * sizeof(float));
    #pragma omp parallel for
    for (int i = 0; i < NUM_SAMPLES; ++i)
    {
        // i < 5: yakın mesafe (DC sızıntısı) → atla
        if (i < 5) continue;

        for (int j = 0; j < NUM_CHIRPS; ++j)
        {
            float temp = 0.0f;
            int in_idx = i * NUM_CHIRPS + j;

            for (int ch = 0; ch < NUM_CHANNELS; ++ch)
                temp += std::norm(all_transposed[ch * TOTAL_SIZE + in_idx]);

            // FFTSHIFT: 0-Doppler'i merkeze kaydır
            int shifted_j = (j + NUM_CHIRPS / 2) % NUM_CHIRPS;
            sumVector[i * NUM_CHIRPS + shifted_j] = temp;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    auto cfar = std::chrono::high_resolution_clock::now();

    cfarData.power = std::vector<float>(sumVector, sumVector + TOTAL_SIZE);
    avxcfar.process(cfarData);
    auto cfar_end = std::chrono::high_resolution_clock::now();
    applyPeakRelativeFilter(cfarData.truth, cfarData.power.data(), NUM_SAMPLES, NUM_CHIRPS);

    fmcwCpuTime = std::chrono::duration<float, std::milli>(end - start).count();
    cfarCpuTime =  std::chrono::duration<float, std::milli>(cfar_end - cfar).count();

    vfcpuTime.push_back(fmcwCpuTime);
    vCfarCpuTime.push_back(cfarCpuTime);
    //if(output.size() != TOTAL_SIZE) output.resize(TOTAL_SIZE);
    //std::memcpy(output.data(), ptroutput, TOTAL_SIZE * sizeof(Complex));
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

float cpu_fmcw::getCpuTimeTotal()
{
    float fSum = 0.0f;
    for(size_t i = 0; i < vfcpuTime.size(); ++i)
    {
        fSum += vfcpuTime[i] + vCfarCpuTime[i];
    }
    return fSum;
}

float cpu_fmcw::getCfarCpuTime()
{
    float fSum = 0.0f;
    for(auto i : vCfarCpuTime)
    {
        fSum += i;
    }
    return fSum/vCfarCpuTime.size();
}

float* cpu_fmcw::getOutput()
{
    return sumVector;
}