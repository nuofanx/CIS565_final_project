#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

__global__ void elementwiseAdd(float* dest, float * src, int size) {
     int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
     if (thread_index < size)
        dest[i] = dest[i] + src[i];
}
