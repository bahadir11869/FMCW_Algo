#pragma once
#include <string>
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>
#include <cstdio> // Dosya islemleri icin
#include <thread> // Sleep için gerekli
#include <filesystem>
#include <math.h>

namespace fs = std::filesystem;

const int NUM_CHIRPS = 2048;      // Slow Time (Y)
const int NUM_SAMPLES = 1024;     // Fast Time (X)
const int TOTAL_SIZE = NUM_CHIRPS * NUM_SAMPLES;
const int TILE_DIM = 32;         // GPU Blok Boyutu

// Radar Fizigi
const float C = 3e8f;
const float FC = 77e9f;
const float BANDWIDTH = 150e6f;
const float CHIRP_DURATION = 50e-6f;
const float SLOPE = BANDWIDTH / CHIRP_DURATION;
#define PI  3.14159265359f

const float range_res = C / (2.0f * BANDWIDTH);
const float wavelength = C / FC;
const float frame_duration = NUM_CHIRPS * CHIRP_DURATION;
const float velocity_res = wavelength / (2.0f * frame_duration);
const float gating_threshold = 3.0f;

using Complex = std::complex<float>;


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

struct stTarget {
    int id;
    float range;    // m
    float velocity; // m/s
    float amplitude;// 0.0 - 1.0
};

template <typename T>
inline void calculate_RMSE_Vectors(const std::vector<T>& cpu_vec, 
                              const std::vector<T>& gpu_vec, 
                              int Nx, int Ny, 
                              bool is_gpu_transposed) {
    
    // Boyut güvenlik kontrolü
    if (cpu_vec.size() != gpu_vec.size()) {
        std::cerr << "UYARI: CPU ve GPU vektor boyutlari esit degil!" << std::endl;
        // İsterseniz burada return -1 yapabilirsiniz ama hesaplamaya devam ediyoruz.
    }

    double total_sq_error = 0.0;
    int total_elements = Nx * Ny;

    for (int r = 0; r < Nx; ++r) {
        for (int c = 0; c < Ny; ++c) {
            
            // 1. CPU İndeksi (Düz / Row-Major)
            int idx_cpu = r * Ny + c;
            
            // 2. GPU İndeksi (Transpoze durumuna göre)
            int idx_gpu;
            if (is_gpu_transposed) {
                // Eğer GPU verisi Transpoze geldiyse (Satırlar Sütun olmuş)
                // Erişim: c * SatırSayısı + r
                idx_gpu = c * Nx + r; 
            } else {
                // GPU verisi de Düz ise
                idx_gpu = r * Ny + c;
            }

            // Değerleri Vektörden Çek
            T val_cpu = cpu_vec[idx_cpu];
            T val_gpu = gpu_vec[idx_gpu];

            // Farkı Hesapla
            // Eğer 'Complex' tipiniz std::complex ise bu operatörler çalışır.
            // Eğer custom struct ise aşağıda elle hesaplama (manual) kısmına bakın.
            T diff = val_cpu - val_gpu; 
            
            // std::norm -> |z|^2 (real^2 + imag^2) döndürür
            total_sq_error += std::norm(diff);
        }
    }

    std::cout <<"RMSE : " << std::sqrt(total_sq_error / total_elements) << std::endl;
}

inline void veriUret( std::vector<Complex>& inputData, std::vector<stTarget>targets)
{
    if (inputData.size() == TOTAL_SIZE)
    {
        for (int i = 0; i < NUM_CHIRPS; ++i)
        {
            for (int j = 0; j < NUM_SAMPLES; ++j) 
            {
                float t = (float)j / NUM_SAMPLES * CHIRP_DURATION;
                float totalTime = i * CHIRP_DURATION + t;
                Complex sum_signal(0.0f, 0.0f);
                for (const auto& tgt : targets) 
                {
                    float tau = 2.0f * (tgt.range + tgt.velocity * totalTime) / C;
                    float phase = 2.0f * PI * (SLOPE * tau * t + FC * tau);
                    sum_signal += std::polar(tgt.amplitude, phase);
                }
                inputData[i * NUM_SAMPLES + j] = sum_signal;
            }
        }
    }
    else
    {
        inputData = {};
        printf("InputData size Yanlis!!!!! \n");
    }
}

inline void dosyayaYaz(const char* cpDosyaIsmi, float fOrtCpuSuresi, float fOrtGpuSuresi, std::vector<Complex> vResult, std::vector<stTarget> targets, float threshold, bool b2DFFTMi = false)
{
    FILE* fp = fopen(cpDosyaIsmi, "w");
    if (fp == NULL) 
    {
        std::cerr << "Dosya acilamadi!" << std::endl;
        return;
    }
    fprintf(fp, "=== RADAR ANALIZ RAPORU ===\n");
    fprintf(fp, "Resolution    : Range = %.4f m | Velocity = %.4f m/s\n", range_res, velocity_res);
    fprintf(fp, "--- HEDEF TESPIT TABLOSU ---\n");
    fprintf(fp, "%-4s | %-12s | %-12s | %-10s || %-12s | %-12s | %-10s | %-8s\n", 
           "ID", "Est_R(m)", "True_R(m)", "Diff_R", "Est_V(m/s)", "True_V(m/s)", "Diff_V", "Match");
    fprintf(fp, "----------------------------------------------------------------------------------------------------\n");
    int detected_count = 0;
    if (!b2DFFTMi)
    {
        for (int r = 1; r < NUM_SAMPLES - 1; ++r) 
        {
            for (int v = 0; v < NUM_CHIRPS; ++v) 
            {
                int idx = r * NUM_CHIRPS + v;
                float amp = std::abs(vResult[idx]);

                if (amp > threshold) {
                    float prev = std::abs(vResult[(r - 1) * NUM_CHIRPS + v]);
                    float next = std::abs(vResult[(r + 1) * NUM_CHIRPS + v]);

                    if (amp >= prev && amp > next) // Lokal maks
                    {
                        detected_count++;
                        
                        float est_r = r * range_res;
                        float est_v = (v < NUM_CHIRPS / 2) ? (v * velocity_res) : ((v - NUM_CHIRPS) * velocity_res);

                        int best_id = -1;
                        float min_norm_error = 1e9f;
                        float true_r_val = 0, true_v_val = 0;

                        for(const auto& t : targets) 
                        {
                            float err_r_bin = std::abs(t.range - est_r) / range_res;
                            float err_v_bin = std::abs(t.velocity - est_v) / velocity_res;
                            float total_error = std::sqrt(err_r_bin * err_r_bin + err_v_bin * err_v_bin);

                            if(total_error < min_norm_error) 
                            {
                                min_norm_error = total_error;
                                best_id = t.id;
                                true_r_val = t.range;
                                true_v_val = t.velocity;
                            }
                        }

                        // Dosyaya Yaz (fprintf kullanarak)
                        if (best_id != -1 && min_norm_error < gating_threshold) {
                            fprintf(fp, "#%-3d | %-12.3f | %-12.3f | %-10.3f || %-12.3f | %-12.3f | %-10.3f | OK(ID:%d)\n", 
                                detected_count, 
                                est_r, true_r_val, est_r - true_r_val, 
                                est_v, true_v_val, est_v - true_v_val, 
                                best_id);
                        } else {
                            fprintf(fp, "#%-3d | %-12.3f | %-12s | %-10s || %-12.3f | %-12s | %-10s | GHOST\n", 
                                detected_count, est_r, "-", "-", est_v, "-", "-");
                        }
                    }
                }
            }
        }
    }
    else
    {
        for (int r = 1; r < NUM_SAMPLES - 1; ++r) 
        {     // Range (Sütunlarda geziyoruz)
            for (int v = 0; v < NUM_CHIRPS; ++v) 
            {      // Velocity (Satırlarda geziyoruz)
                
                // FORMÜL DEĞİŞİKLİĞİ:
                // Manuel Transpose'da: idx = r * NUM_CHIRPS + v idi.
                // 2D FFT Düzeninde   : idx = v * NUM_SAMPLES + r dir.
                int idx = v * NUM_SAMPLES + r; 

                float amp = std::abs(vResult[idx]);

                if (amp > threshold) 
                {
                    // Peak Finding (Range ekseninde komşulara bakıyoruz)
                    // Range, bellek düzeninde "Sütun" olduğu için komşular +1 ve -1 indeksindedir.
                    // (Manuel transpose kodunda komşular +NUM_CHIRPS ve -NUM_CHIRPS uzaklıktaydı çünkü Range satırdı)
                    
                    int prev_idx = v * NUM_SAMPLES + (r - 1);
                    int next_idx = v * NUM_SAMPLES + (r + 1);

                    float prev = std::abs(vResult[prev_idx]);
                    float next = std::abs(vResult[next_idx]);

                    if (amp >= prev && amp > next) {
                        detected_count++;
                        
                        // --- Fiziksel Değer Hesaplama ---
                        float est_r = r * range_res;
                        
                        // Doppler Hız Hesabı (+/- Hız ayırımı)
                        float est_v;
                        if (v < NUM_CHIRPS / 2) 
                        {
                            est_v = v * velocity_res; // Pozitif Hız (Uzaklaşan)
                        } else 
                        {
                            est_v = (v - NUM_CHIRPS) * velocity_res; // Negatif Hız (Yaklaşan)
                        }

                        // --- Eşleştirme (Nearest Neighbor) ---
                        // Bu kısım diğer kodla birebir aynıdır
                        int best_id = -1;
                        float min_norm_error = 1e9f;
                        float true_r_val = 0, true_v_val = 0;

                        for(const auto& t : targets) 
                        {
                            float err_r_bin = std::abs(t.range - est_r) / range_res;
                            float err_v_bin = std::abs(t.velocity - est_v) / velocity_res;
                            
                            // Öklid mesafesi
                            float total_error = std::sqrt(err_r_bin * err_r_bin + err_v_bin * err_v_bin);

                            if(total_error < min_norm_error) {
                                min_norm_error = total_error;
                                best_id = t.id;
                                true_r_val = t.range;
                                true_v_val = t.velocity;
                            }
                        }

                        // Yazdirma
                        if (best_id != -1 && min_norm_error < gating_threshold) 
                        {
                            fprintf(fp, "#%-3d | %-12.3f | %-12.3f | %-10.3f || %-12.3f | %-12.3f | %-10.3f | OK(ID:%d)\n", 
                                detected_count, 
                                est_r, true_r_val, est_r - true_r_val, 
                                est_v, true_v_val, est_v - true_v_val, 
                                best_id);
                        } else 
                        {
                            fprintf(fp, "#%-3d | %-12.3f | %-12s | %-10s || %-12.3f | %-12s | %-10s | GHOST\n", 
                                detected_count, est_r, "-", "-", est_v, "-", "-");
                        }
                    }
                }
            }
        }

    }

    
    fprintf(fp, "----------------------------------------------------------------------------------------------------\n");
    fclose(fp); // Dosyayı kapatmayı unutma!

}


inline void save_rdm_data(const char* filename, const std::vector<Complex>& gpu_result, bool is_input_transposed)
{
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        std::cerr << "Dosya acilamadi: " << filename << std::endl;
        return;
    }

    // Hedef: Python imshow için standart format: [Doppler(Satır)][Range(Sütun)]
    // Bu yüzden dış döngü Doppler (Chirp), iç döngü Range (Sample) olmalı.
    
    for (int v = 0; v < NUM_CHIRPS; ++v)      // Satırlar: Hız (Doppler)
    {
        for (int r = 0; r < NUM_SAMPLES; ++r) // Sütunlar: Mesafe (Range)
        {
            int idx;
            
            if (is_input_transposed) 
            {
                // DURUM 1: Manuel Transpose Çıktısı [Sample][Chirp] şeklindedir.
                // Biz (v, r) koordinatına ulaşmak için bellekte (r, v) konumuna gitmeliyiz.
                // Bellek formülü: r * (SatırUzunluğu=NUM_CHIRPS) + v
                idx = r * NUM_CHIRPS + v; 
            } 
            else 
            {
                // DURUM 2: 2D FFT Çıktısı [Chirp][Sample] şeklindedir (Orijinal düzen).
                // Bellek formülü: v * (SatırUzunluğu=NUM_SAMPLES) + r
                idx = v * NUM_SAMPLES + r;
            }

            // Genlik hesapla
            float amp = std::abs(gpu_result[idx]);
            
            fprintf(fp, "%.4f", amp);
            if (r < NUM_SAMPLES - 1) fprintf(fp, ",");
        }
        fprintf(fp, "\n"); // Satır bitti, alta geç
    }

    fclose(fp);
}