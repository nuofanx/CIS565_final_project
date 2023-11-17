

#pragma once

#include <cuda_runtime.h>
#include <cub/cub.cuh>

// Global functions are also called "kernels". It's the functions that you may call from the host side using CUDA kernel call semantics (<<<...>>>).
// Device functions can only be called from other device or global functions. __device__ functions cannot be called from host code.
__device__ void softmax_gpu(void* __restrict__ x, int size){
    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_val;

    int tid = threadIdx.x;
    int step = blockDim.x;

    // find max value (for numerical stability)
    float max_val = tid < size ? x[tid] : 0;
    for (int i = tid + step; i < size; i += step)
        if (x[i] > max_val)
            max_val = x[i];

    max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
    if (threadIdx.x == 0)
        shared_val = max_val;
    __syncthreads();
    max_val = shared_val;

    // exp and sum
    float sum = 0.0f;
    for (int i = tid; i < size; i += step) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    sum = BlockReduce(temp).Sum(sum);
    if (threadIdx.x == 0)
        shared_val = sum;
    __syncthreads();
    sum = shared_val;

    // normalize
    for (int i = tid; i < size; i += step)
        x[i] /= sum;
}

#define MAX_SEQ_LEN 8192
__global__ void multiHeadAttentionKernel_naive(void* output, void* sq, void* key, void* value, int num_heads, int head_size, int loff, int seq_len, int dim){
    int h = blockIdx.x;
    // get Q vector based on the address of sq and head index for current thread
    half * q = sq + h * head_size;
    // get attention scores for this head
    __shared__ float att[MAX_SEQ_LEN];
    // iterate over all timesteps, including the current one
    for (int t = threadIdx.x; t < seq_len; t+= blockDim.x) {
        // get the key vector for this head and at this timestep
        const float* k = key + loff + t * dim + h * head_size;
        // calculate the attention score as the dot product of q and k

        float score = 0.0f;
        for (int i = 0; i < head_size; i++)
            score += (float)q[i] * (float)k[i];
        score /= sqrtf(head_size);
        // save the score to the attention buffer
        att[t] = score;
    }
    __syncthreads();
    // softmax the scores to get attention weights
    softmax_gpu(att, seq_len);
    __syncthreads();

    // calculate weighted sum of the values, store back into xb
    for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
        float val = 0.0f;
        for (int t = 0; t < seq_len; t++)
            val += att[t] * value[loff + t * dim + h * head_size + i];
        output[h * head_size + i] = val;
    }
}
