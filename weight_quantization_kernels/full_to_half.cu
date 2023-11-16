
#pragma once

#include <cuda_runtime.h>

__global__ void ConvertFP32toFP16(half *out, float *in, int elements) {
    int index = blockIdx.x * 256 + threadIdx.x;
    if (index < elements)
        out[index] = (half)in[index];
}