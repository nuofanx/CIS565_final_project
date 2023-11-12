#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// kernel function for x * sigma(x)
__global__ void siluKernel(float* dest, float* src, int size){
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_index < size){
        // extract val
        val = dest[i]
        // sigma(x)
        val *= 1.0f / (1.0f + expf(-val));
        // x * sigma(x)
        val *= src[i];
    }
}
