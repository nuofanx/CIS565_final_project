#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

__global__ void elementwiseAdd(float* dest, float * src, int size, int weight_quant_num) {
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_index < size){
        switch (weight_quant_num){
            case 0:
                dest[i] = dest[i] + src[i];
                break;
            case 1:
                dest[i] = (half)((float)dest[i] + (float)src[i]);
                break;
            default:
                throw std::invalid_argument("Unknown weight quantization number");
        }
    }
}