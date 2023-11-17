
#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// An important point to bear in mind is that when two consecutive load operations are carried out on the same addresses, the second operation is likely to get the data from the GPU cache, meaning it doesn't cost double DDR loading to perform two passes within the same Triton program.
// For instance, the reduction operation (sum) is executed outside the loop due to its cost (this CUDA presentation: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf will give you a basic understanding of how complicated reductions are at warp level).
// single block 
__global__ void rmsNormKernel(float* o, float* x, float* weight, int size, int elementsPerThread){
    //  first loop over input tensor to compute the root mean of the square
    float ss = 0.0f;
    for (int i = 0; i < elementsPerThread; i++) {
        int index = threadIdx.x + i * 1024;
        if (index < size)
            ss += (float) x[index];
    }

    // calculate result and load to shared memory 
    __shared__ float shared_ss;
    // take mean, add eps, then sqrt 
    if (threadIdx.x == 0) {
        ss /= size;
        ss += 1e-5f;
        ss = 1.0f / sqrtf(ss);
        shared_ss = ss;
    }
    __syncthreads();
    // read from shared memory 
    ss = shared_ss;
    //  we keep this reduction operation outside the loop for perf reasons
    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage temp;
    ss = BlockReduce(temp).Sum(ss * ss);

    //  apply the normalization and multiply by RMS weights
    for (int i = 0; i < elementsPerThread; i++) {
        int index = threadIdx.x + i * 1024;
        if (index < size) {
            float val = (float)x[index];
            val *= ss * (float)weight[index];
            o[index] = (half)val;
        }
    }
}
