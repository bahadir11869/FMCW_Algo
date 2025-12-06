// Dosya Adı: Kernels.h
#pragma once
#include "defines.h"

// Kernel Prototipi (Gövdesi yok, sadece imzası)
__global__ void transpose_optimized_kernel(cuComplex* input, cuComplex* output, int width, int height);