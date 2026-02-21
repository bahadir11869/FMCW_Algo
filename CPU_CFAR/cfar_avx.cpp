#include "cfar_avx.hpp"
#include <immintrin.h>

AVXCFAR::AVXCFAR(CFARParams params): P(params) 
{
    update_params(params);
}


void AVXCFAR::update_params(CFARParams params)
{
        P = params;
        int win_r = P.ref_r + P.guard_r;
        int win_c = P.ref_c + P.guard_c;
        int guard_r = P.guard_r;
        int guard_c = P.guard_c;

        int big_area = (2 * win_r + 1) * (2 * win_c + 1);
        int guard_area = (2 * guard_r + 1) * (2 * guard_c + 1);
        N_ref = big_area - guard_area;

        // Alpha hesabı (CA-CFAR)
        alpha =  N_ref  *  (std::pow(P.pfa, -1.0f / N_ref) - 1.0f);
        alpha /=  (float)N_ref;  
        // Çarpma işlemini hızlandırmak için bölme işlemini peşinen yapıyoruz
}

void AVXCFAR::compute_sat(const float* power, float* sat, int rows, int cols) 
{
    // 1. Satır bazlı kümülatif toplam
    #pragma omp parallel for
    for (int r = 0; r < rows; ++r) {
        float acc = 0.0f;
        int row_offset = r * cols;
        for (int c = 0; c < cols; ++c) {
            acc += power[row_offset + c];
            sat[row_offset + c] = acc;
        }
    }    
        // 2. Sütun bazlı kümülatif toplam
        // Cache dostu olması için bloklama yapılabilir ama basitlik için direkt geçiyoruz
        // Sütun erişimi CPU'da yavaştır, bu yüzden transpose mantığı düşünülebilir
        // Ancak SAT bir kere hesaplanır, detection milyon kere yapılır.
        
        // Hız için sütunları da parallel yapalım ama cache thrashing riski var.
        // Güvenli yöntem: Tek thread veya bloklu yapı.
        // Basit ve hızlı bir yöntem (Cache dostu değil ama paralel):
        
        #pragma omp parallel for
        for (int c = 0; c < cols; ++c) {
            float acc = 0.0f;
            for (int r = 0; r < rows; ++r) {
                acc += sat[r * cols + c];
                sat[r * cols + c] = acc;
            }
        }
}


void AVXCFAR::process(CFARData& d) 
{
    const int rows = P.rows;
    const int cols = P.cols;
    
    // SAT için bellek ayır (veya dışarıdan al)
    std::vector<float> sat_vec(rows * cols);
    float* sat = sat_vec.data();
    
    // 1. SAT Hesapla
    compute_sat(d.power.data(), sat, rows, cols);

    // Pencere boyutları
    const int win_r = P.ref_r + P.guard_r;
    const int win_c = P.ref_c + P.guard_c;
    const int gr = P.guard_r;
    const int gc = P.guard_c;
    
    // AVX sabiti: Alpha Normalized
    __m256 v_alpha = _mm256_set1_ps(alpha);

    // 3. Sliding Window (AVX ile Hızlandırılmış)
    // Kenarlardan (margin) kaçınarak döngü kuruyoruz
    
    #pragma omp parallel for schedule(dynamic)
    for (int r = win_r; r < rows - win_r; ++r) {
        
        // Satır pointerları (SAT tablosunda hızlı erişim için)
        // Big Window Köşeleri (A: TopLeft, B: TopRight, C: BotLeft, D: BotRight)
        // İsimlendirme SAT standardına göre: D + A - B - C
        // Koordinatlar: (r2, c2), (r1-1, c2), (r2, c1-1), (r1-1, c1-1)
        
        // Outer (Big) Window Y koordinatları
        int r1_out = r - win_r - 1; 
        int r2_out = r + win_r;
        
        // Inner (Guard) Window Y koordinatları
        int r1_in = r - gr - 1;
        int r2_in = r + gr;

        // Satır ofsetleri (pointer aritmetiği)
        const float* p_out_top = (r1_out >= 0) ? &sat[r1_out * cols] : nullptr;
        const float* p_out_bot = &sat[r2_out * cols];
        const float* p_in_top  = (r1_in >= 0)  ? &sat[r1_in * cols]  : nullptr;
        const float* p_in_bot  = &sat[r2_in * cols];
        
        const float* p_power_row = &d.power[r * cols];
        bool* p_truth_row = &d.truth[r * cols];

        int c = win_c;
        
        // AVX Döngüsü: Her adımda 8 sütun işler
        for (; c <= cols - win_c - 8; c += 8) {
            
            // --- 1. Big Window Toplamı ---
            // X Koordinatları (8 eleman için vektör ofsetleri)
            // c2 = c + win_c (Sağ kenar)
            // c1 = c - win_c (Sol kenar) -> SAT formülünde (c1 - 1) kullanılır
            
            int idx_right_out = c + win_c;
            int idx_left_out  = c - win_c - 1;

            // D Köşesi (Bottom-Right)
            __m256 D_out = _mm256_loadu_ps(&p_out_bot[idx_right_out]);
            
            // C Köşesi (Bottom-Left)
            __m256 C_out = (idx_left_out >= 0) ? _mm256_loadu_ps(&p_out_bot[idx_left_out]) : _mm256_setzero_ps();

            // B Köşesi (Top-Right)
            __m256 B_out = (p_out_top) ? _mm256_loadu_ps(&p_out_top[idx_right_out]) : _mm256_setzero_ps();

            // A Köşesi (Top-Left)
            __m256 A_out = (p_out_top && idx_left_out >= 0) ? _mm256_loadu_ps(&p_out_top[idx_left_out]) : _mm256_setzero_ps();

            // Sum Big = D - C - B + A
            __m256 sum_big = _mm256_sub_ps(D_out, C_out);
            sum_big = _mm256_sub_ps(sum_big, B_out);
            sum_big = _mm256_add_ps(sum_big, A_out);


            // --- 2. Guard Window Toplamı ---
            int idx_right_in = c + gc;
            int idx_left_in  = c - gc - 1;

            __m256 D_in = _mm256_loadu_ps(&p_in_bot[idx_right_in]);
            __m256 C_in = (idx_left_in >= 0) ? _mm256_loadu_ps(&p_in_bot[idx_left_in]) : _mm256_setzero_ps();
            __m256 B_in = (p_in_top) ? _mm256_loadu_ps(&p_in_top[idx_right_in]) : _mm256_setzero_ps();
            __m256 A_in = (p_in_top && idx_left_in >= 0) ? _mm256_loadu_ps(&p_in_top[idx_left_in]) : _mm256_setzero_ps();

            __m256 sum_guard = _mm256_sub_ps(D_in, C_in);
            sum_guard = _mm256_sub_ps(sum_guard, B_in);
            sum_guard = _mm256_add_ps(sum_guard, A_in);

            // --- 3. Eşik Hesabı ---
            // Ref = Big - Guard
            __m256 v_ref = _mm256_sub_ps(sum_big, sum_guard);
            // Threshold = Ref * Alpha_Norm
            __m256 v_thr = _mm256_mul_ps(v_ref, v_alpha);

            // --- 4. Karşılaştırma ---
            // CUT (Cell Under Test) Power değerini yükle
            __m256 v_cut = _mm256_loadu_ps(&p_power_row[c]);

            // CUT > Thr ?
            __m256 v_cmp = _mm256_cmp_ps(v_cut, v_thr, _CMP_GT_OQ);

            // --- 5. Sonucu Yazma ---
            // Bool array olduğu için maskeyi int'e çevirip tek tek yazacağız
            int mask = _mm256_movemask_ps(v_cmp);
            
            if (mask) { // Sadece tespit varsa yaz (Branch Prediction avantajı)
                // Loop unrolling ile hızlı yazım
                if (mask & 1) p_truth_row[c + 0] = true;
                if (mask & 2) p_truth_row[c + 1] = true;
                if (mask & 4) p_truth_row[c + 2] = true;
                if (mask & 8) p_truth_row[c + 3] = true;
                if (mask & 16) p_truth_row[c + 4] = true;
                if (mask & 32) p_truth_row[c + 5] = true;
                if (mask & 64) p_truth_row[c + 6] = true;
                if (mask & 128) p_truth_row[c + 7] = true;
            }
        }

        // Cleanup Loop (Kalan sütunlar için scalar işlem)
        // Eğer sütun sayısı 8'in katı değilse son birkaç piksel burada işlenir
        for (; c < cols - win_c; ++c) {
            float big = sat_rect_sum(sat, rows, cols, r, c, win_r, win_c);
            float guard = sat_rect_sum(sat, rows, cols, r, c, gr, gc);
            float ref = big - guard;
            float thr = ref * alpha;
            if (d.power[r * cols + c] > thr) {
                d.truth[r * cols + c] = true;
            }
        }
    }
}


float AVXCFAR::sat_rect_sum(const float* sat, int rows, int cols, int r, int c, int wr, int wc) 
{
        int r1 = r - wr - 1; int c1 = c - wc - 1;
        int r2 = r + wr;     int c2 = c + wc;
        
        float D = sat[r2 * cols + c2];
        float C = (c1 >= 0) ? sat[r2 * cols + c1] : 0.0f;
        float B = (r1 >= 0) ? sat[r1 * cols + c2] : 0.0f;
        float A = (r1 >= 0 && c1 >= 0) ? sat[r1 * cols + c1] : 0.0f;
        return D - C - B + A;
}