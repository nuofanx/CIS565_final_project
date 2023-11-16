#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// One head per block
// https://ai.lefebvre-sarrut.eu/2023/07/20/deep-dive-into-kernel-fusion-accelerating-inference-in-llama-v2/#rewriting-without-complex-number-arithmetic
// kernel fusion reduces global memory load/store operations
// This savings can be very significant for memory-bound operations on the GPU. 
// the overall performance improvement is usually proportional to the reduction in number of load/store operations.

__global__ void RoPERotationKernel(float* sq, float* sk, float* f_real, float* f_imag, int num_heads, int head_size){
    int h = blockIdx.x;
    // splits the input tensors sq and sk into real and imaginary parts
    // locate the correct pointer using head_size
    float* q = sq + h * head_size;
    float* k = sk + h * head_size;

    int i = threadIdx.x * 2;
    // find the correct index 
    float q0 = q[i];
    float q1 = q[i + 1];
    float k0 = k[i];
    float k1 = k[i + 1];
    float fcr = f_real[i / 2];
    float fci = f_imag[i / 2];
    //  Perform the equivalent of complex number multiplication 
    q[i] = q0 * fcr - q1 * fci;
    q[i + 1] = q0 * fci + q1 * fcr;
    k[i] = k0 * fcr - k1 * fci;
    k[i + 1] = k0 * fci + k1 * fcr;
}