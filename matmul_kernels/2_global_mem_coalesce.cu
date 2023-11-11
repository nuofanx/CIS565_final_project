#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

__global__ void matmulKernel_gmc(float* C, float* A, float* B, int M, int N, int K){
    // coalescing 
    const int x = blockIdx.x * blockDim.x + (threadIdx.x / blockDim.x);
    const int y = blockIdx.y * blockDim.y + (threadIdx.x % blockDim.x);

    // loop through each element that needs to be computed 
    if (x< M && y < N){
        float tmp = 0.0f;

        for (int i = 0; i < K; i++) {
            // indexing into strided in-memory representations of matrices.
            tmp += A[x * K + i] * B[i* N + y];
        }
        C[x*K + y] = tmp;
    }
}
