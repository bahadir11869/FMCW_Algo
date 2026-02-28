#pragma once
#include "../defines.h"
#include "../CPU_CFAR/cfar_cpu.hpp"
#include "../CPU_CFAR/cfar_avx.hpp"
#include <mkl.h>

class cpu_fmcw
{
private:
    void cpu_recursive_fft(std::vector<Complex>& a); 
    float fmcwCpuTime;
    float cfarCpuTime;

    std::vector<float> vfcpuTime;
    std::vector<float> vCfarCpuTime;
    std::vector<Complex> output;
    DFTI_DESCRIPTOR_HANDLE handRange;
    DFTI_DESCRIPTOR_HANDLE handDoppler;
    Complex* all_transposed;
    float* sumVector;
    CFARParams p;
    CPUCFAR cpu_sat;
    AVXCFAR avxcfar;


public:
    CFARData cfarData;
    cpu_fmcw();
    ~cpu_fmcw();
    void run_cpu_basic(const std::vector<Complex>& input);
    void run_cpu_openmp(const std::vector<Complex>& input);
    void run_cpu_avx(Complex* input, Complex* ptroutput);

    float getCpuTime();
    float getCpuTimeTotal();
    float getCfarCpuTime();
    float* getOutput();
};
