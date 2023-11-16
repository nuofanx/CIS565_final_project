

#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define MAX_SEQ_LEN 8192
__global__ void multiHeadAttentionKernel_naive(float* output, float* sq, float* key, float* value, int num_heads, int head_size, int loff, int seq_len, int dim){
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
