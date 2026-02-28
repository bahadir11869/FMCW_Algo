// nvcc -arch=sm_86 -std=c++17 -O3 mainEval.cu --options-file compile2.txt -o FMCW_Eval.exe
//
// Cikti CSV sutunlari:
//   frame_id : .bin dosya adi (ör. 000006)
//   range_m  : mesafe (metre)
//   vel_ms   : hiz (m/s), pozitif = uzaklasan, negatif = yaklasan
//   power    : ham guc degeri
//   group    : MOVING / STATIC

#include "defines.h"
#include "GPU_FMCW/gpu_fmcw.h"
#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>

int main()
{
    const std::string BIN_DIR    = "matlab2bin/2019_04_09_bms1000/radar_bin";
    const std::string OUTPUT_CSV = "eval_detections.csv";

    // ── 1. Tum .bin dosyalarini sirali olarak topla ──────────────────────────
    std::vector<std::string> binFiles;
    for (const auto& e : fs::directory_iterator(BIN_DIR))
        if (e.path().extension() == ".bin")
            binFiles.push_back(e.path().string());
    std::sort(binFiles.begin(), binFiles.end());

    const int totalFrames = (int)binFiles.size();
    printf("Toplam %d frame isleniyor...\n", totalFrames);

    // ── 2. GPU nesnesi ve bellek (CUDA init bir kez yapilir) ─────────────────
    gpu_fmcw gpu(2, "GPU_FMCW/eval_timing.txt");   // 2 = cuFFT 2DFFT

    std::vector<Complex> fullData(TOTAL_ELEMENTS);
    Complex* h_input;
    float*   h_output;
    gpuErrchk(cudaMallocHost(&h_input,  TOTAL_ELEMENTS * sizeof(Complex)));
    gpuErrchk(cudaMallocHost(&h_output, TOTAL_SIZE     * sizeof(float)));

    // ── 3. Cikti CSV ─────────────────────────────────────────────────────────
    FILE* fp = fopen(OUTPUT_CSV.c_str(), "w");
    if (!fp) { fprintf(stderr, "CSV acilamadi!\n"); return 1; }
    fprintf(fp, "frame_id,range_m,vel_ms,power,group\n");

    // ── 4. Frame dongusu ─────────────────────────────────────────────────────
    // Bellek duzeni (fftshift sonrasi): [Range][Doppler]
    //   indeks = r * NUM_CHIRPS + v
    //   v = NUM_CHIRPS/2  →  sifir Doppler
    const int   CENTER    = NUM_CHIRPS / 2;
    const float RANGE_RES = range_res;
    const float VEL_RES   = velocity_res;

    int processed = 0;
    for (const auto& binPath : binFiles)
    {
        std::string frame_id = fs::path(binPath).stem().string();

        readBin(binPath, fullData);
        memcpy(h_input, fullData.data(), TOTAL_ELEMENTS * sizeof(Complex));

        gpu.run_gpu_2DFFT(h_input, h_output);
        gpuErrchk(cudaDeviceSynchronize());

        // Peak-relative filter: kapat/ac icin APPLY_FILTER tanimla
#ifdef APPLY_FILTER
        applyPeakRelativeFilter(gpu.bpCFAR, gpu.getOutput(), NUM_SAMPLES, NUM_CHIRPS);
#endif

        // Tespitleri yaz
        for (int r = 0; r < NUM_SAMPLES; ++r) {
            for (int v = 0; v < NUM_CHIRPS; ++v) {
                int idx = r * NUM_CHIRPS + v;
                if (!gpu.bpCFAR[idx]) continue;

                float range_m = r * RANGE_RES;
                int   doff    = v - CENTER;
                float vel_ms  = doff * VEL_RES;
                float pwr     = gpu.getOutput()[idx];
                const char* grp = (std::abs(doff) > 2) ? "MOVING" : "STATIC";

                fprintf(fp, "%s,%.2f,%.4f,%.4e,%s\n",
                        frame_id.c_str(), range_m, vel_ms, pwr, grp);
            }
        }

        ++processed;
        if (processed % 100 == 0 || processed == totalFrames)
            printf("  [%d/%d] %s\n", processed, totalFrames, frame_id.c_str());
    }

    fclose(fp);
    printf("Tamamlandi → %s\n", OUTPUT_CSV.c_str());

    cudaFreeHost(h_input);
    cudaFreeHost(h_output);
    return 0;
}
