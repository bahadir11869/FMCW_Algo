#pragma once
#include "cfar_common.hpp"

class CPUCFAR {
public:
    explicit CPUCFAR(const CFARParams& p) : P(p) {}
    CFARStats process(const CFARData& d); // end-to-end: SAT + detect (+istatistik)

private:
    CFARParams P;
};
