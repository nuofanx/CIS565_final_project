#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// created a template function with BLOCKSIZE as its template parameter. 
template <const uint BLOCKSIZE>
__global__ void matmulKernel_global_mem_coalesce(void* C, void* A, void* B, int M, int N, int K, int weight_quant_num){
    // coalescing 
    const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);


    // loop through each element that needs to be computed 
    if (x< M && y < N){
        float tmp = 0.0f;

        for (int i = 0; i < K; i++) {
            // indexing into strided in-memory representations of matrices.
            tmp += (float)A[x * K + i] * (float)B[i* N + y];
        }
        switch (weight_quant_num){
            case 0:
                C[x*K + y] = tmp;
                break;
            case 1:
                C[x*K + y] = (half)tmp;
                break;
            default:
                throw std::invalid_argument("Unknown weight quantization number");
        }
    }
}
