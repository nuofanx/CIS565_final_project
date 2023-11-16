#pragma once

#include <cuda_runtime.h>

__global__ void ConvertFP16toFP32(float* out, half* in, int elements) {
    int index = blockIdx.x * 256 + threadIdx.x;
    if (index < elements)
        out[index] = (float)in[index];
}
