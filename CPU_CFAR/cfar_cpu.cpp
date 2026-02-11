#include "cfar_cpu.hpp"
#include <omp.h>

CFARStats CPUCFAR::process(const CFARData& d) {
    CFARStats s{};

    const int rows = P.rows, cols = P.cols;
    const int win_r = P.ref_r + P.guard_r;
    const int win_c = P.ref_c + P.guard_c;
    const int big_h = 2 * win_r + 1, big_w = 2 * win_c + 1;
    const int guard_h = 2 * P.guard_r + 1, guard_w = 2 * P.guard_c + 1;
    const int N = big_h * big_w - guard_h * guard_w;
    
    const float alpha = N * (std::pow(P.pfa, -1.0f / N) - 1.0f);

    // SAT
    std::vector<float> sat((size_t)rows * cols);
#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r) {
        float acc = 0.f;
        for (int c = 0; c < cols; ++c) {
            acc += d.power[idx(r, c, cols)];
            sat[idx(r, c, cols)] = acc;
        }
    }
#pragma omp parallel for schedule(static)
    for (int c = 0; c < cols; ++c) {
        float acc = 0.f;
        for (int r = 0; r < rows; ++r) {
            acc += sat[idx(r, c, cols)];
            sat[idx(r, c, cols)] = acc;
        }
    }

    auto rect_sum = [&](int r1, int c1, int r2, int c2)->float {
        float A = sat[idx(r2, c2, cols)];
        float B = (r1 > 0) ? sat[idx(r1 - 1, c2, cols)] : 0.0f;
        float C = (c1 > 0) ? sat[idx(r2, c1 - 1, cols)] : 0.0f;
        float D = (r1 > 0 && c1 > 0) ? sat[idx(r1 - 1, c1 - 1, cols)] : 0.0f;
        return A - B - C + D;
    };

#pragma omp parallel for collapse(2) , schedule(static)
    for (int r = win_r; r < rows - win_r; ++r) {
        for (int c = win_c; c < cols - win_c; ++c) {
            float big = rect_sum(r - win_r, c - win_c, r + win_r, c + win_c);
            float guard = rect_sum(r - P.guard_r, c - P.guard_c, r + P.guard_r, c + P.guard_c);
            float ref = big - guard;
            float thr = alpha * (ref / N);
            float cur = d.power[idx(r, c, cols)];
            //printf("thr:%f, cur:%f\n", thr, cur);
            if(cur > thr) {
                d.truth[idx(r, c, cols)] = true;
                printf("column: %d, row:   %d idx: %d\n", c, r, idx(r, c, cols));
            }
            else{
                
                d.truth[idx(r, c, cols)] = false;
                
            }
        }
    }

    // istatistikler
    /*long long total_targets = 0;
    for (auto v : d.truth) if (v) total_targets++;
    long long tested = (rows - 2 * win_r) * (cols - 2 * win_c);
    long long non = tested - total_targets;
    s.detections = det; s.tp = tp; s.fp = fp;
    s.total_targets = total_targets; s.tested = tested; s.non_targets = non;
    s.Pd = total_targets ? (double)tp / total_targets : 0.0;
    s.Pfa_emp = non ? (double)fp / non : 0.0;

    s.wall_ms = s.cpu_ms_total; // CPU i�in wall=total*/
    return s;
}
