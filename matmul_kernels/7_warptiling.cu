
#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// Warptiling
// Blocktiling: Different blocks can execute in parallel on different SMs.
// Warptiling: Different warps can execute in parallel on different warp schedulers, and concurrently on the same warp scheduler.
// Threadtiling: (a very limited amount of) instructions can execute in parallel on the same CUDA cores (= instruction-level parallelism aka ILP).
// dotIdx loops over contents of SMEM

__global__ void matmulKernel_Warptiling(){
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
        // populate registers for this thread's part of the warptile
        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
            for (uint i = 0; i < TM; ++i) {
            regM[wSubRowIdx * TM + i] =
                As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
                    threadRowInWarp * TM + i];
            }
        }
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            for (uint i = 0; i < TN; ++i) {
            regN[wSubColIdx * TN + i] =
                Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
                    threadColInWarp * TN + i];
            }
        }
    }

    // execute warptile matmul. Later this will map well to
    // warp-wide matrix instructions, executed on tensor cores.
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            // calculate per-thread results with register-cache locality
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                                (wSubColIdx * TN) + resIdxN] +=
                    regM[wSubRowIdx * TM + resIdxM] *
                    regN[wSubColIdx * TN + resIdxN];
                }
            }
        }
    }
}