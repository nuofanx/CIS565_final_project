#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>


// A is input matrix, B is weight matrix 
// matmul operation on matrix A of shape (M, K) and matrix B of shape (N, K)
// resulting matrix C is of shape (M, N)
__global__ void matmulKernel_naive(float* C, float* A, float *B, int M, int N, int K){
    // use strided in-memory representation index x and y 
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    // check if current thread is used in the calculation 
    if (x< M && y < N){
        // init tmp val 
        float tmp = 0.0f;
        // loop through each col of A and row of B, with K elements  
        for (int i = 0; i < K; i++) {
            // x is row index of A and C and y is col index of B and C
            tmp += A[x * K + i] * B[i* N + y];
        }
        // store the result to C
        C[x*K + y] = tmp;
    }
}


