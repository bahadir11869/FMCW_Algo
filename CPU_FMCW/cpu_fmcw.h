#pragma once
#include "../defines.h"

class cpu_fmcw
{
private:
    void cpu_recursive_fft(std::vector<Complex>& a); 
    float fcpuTime; 
    std::vector<float> vfcpuTime;
    std::string strDosyaAdi;
    std::vector<Complex> output;


public:
    cpu_fmcw(std::string strDosyaAdi);
    ~cpu_fmcw();
    void run_cpu_basic(const std::vector<Complex>& input);
    void run_cpu_openmp(const std::vector<Complex>& input);
    void run_cpu_avx(Complex* input, Complex* ptroutput);

    float getCpuTime();
    std::string getFileName();
    std::vector<Complex> getOutput();
};

