#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

template <const int BM, const int BN, const int BK, const int TM, const int TN>
// Vectorize SMEM and GMEM Accesses
__global__ void matmulKernel_vecSMEM_GMEM(void* C, void* B, void* A, int M, int N, int K, int weight_quant_num){
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    // BN/TN are the number of threads to span a column
    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    // allocate space for the current blocktile in smem
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Move blocktile to beginning of A's row and B's column
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // allocate thread-local cache for results in registerfile
    float threadResults[TM * TN] = {0.0};
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    
    // calculating the indices that this thread will load into SMEM
    // we'll load 128bit / 32bit = 4 elements per thread at each step
    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);

      // outer-most loop over block tiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        float4 tmp =
            reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
        // transpose A during the GMEM to SMEM transfer
        As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
        As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
        As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
        As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

        //why faster than just manually unrolling the access (or using pragma unroll)
        reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] =
            reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];
        __syncthreads();

        // the compiler has no way to verify that the float* B pointer that is passed to the kernel is 128b aligned, which would be a requirement for using LDG.E.128. So the reinterpret_cast’s only purpose is to promise the compiler that the float* B pointer will be aligned.
        // Bs[innerRowB * BN + innerColB * 4 + 0] = B[innerRowB * N + innerColB * 4 + 0];
        // Bs[innerRowB * BN + innerColB * 4 + 1] = B[innerRowB * N + innerColB * 4 + 1];
        // Bs[innerRowB * BN + innerColB * 4 + 2] = B[innerRowB * N + innerColB * 4 + 2];
        // Bs[innerRowB * BN + innerColB * 4 + 3] = B[innerRowB * N + innerColB * 4 + 3];

        // advance blocktile
        A += BK;     // move BK columns to right
        B += BK * N; // move BK rows down

        // calculate per-thread results
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
        // block into registers
        for (uint i = 0; i < TM; ++i) {
            regM[i] = As[dotIdx * BM + threadRow * TM + i];
        }
        for (uint i = 0; i < TN; ++i) {
            regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
        }
        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
            for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            threadResults[resIdxM * TN + resIdxN] +=
                regM[resIdxM] * regN[resIdxN];
            }
        }
        }
        __syncthreads();
    }

    // write out the results
    for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
        // load C vector into registers
        float4 tmp = reinterpret_cast<float4 *>(
            &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
        // perform GEMM update in reg
        tmp.x = alpha * threadResults[resIdxM * TN + resIdxN] + beta * tmp.x;
        tmp.y = alpha * threadResults[resIdxM * TN + resIdxN + 1] + beta * tmp.y;
        tmp.z = alpha * threadResults[resIdxM * TN + resIdxN + 2] + beta * tmp.z;
        tmp.w = alpha * threadResults[resIdxM * TN + resIdxN + 3] + beta * tmp.w;
        // write back
        reinterpret_cast<float4 *>(
            &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] =
            tmp;
        }
    }
}