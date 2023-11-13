#pragma once

#include <stdio.h>
#include <stdlib.h> 
#include <cuda_runtime.h>
#include <cub/cub.cuh>

//one output per warp so that we can parallelize the dot product across the warp
__global__ void matmulKernel_cubreduce(void* output, void* input, void* weight, int input_dim, int output_dim, int hidden_dim, int weight_quant_num) {
    int index = blockIdx.x * blockDim.y + threadIdx.y;
    if (index >= output_dim)
        return;

    float sum = 0;
    for (int i = 0; i < hidden_dim; i++) {
        int j = i * 32 + threadIdx.x;
        if (j < input_dim)
            sum += ((float) weight[index * input_dim + j]) * ((float)input[j]);
    }

    using WarpReduce = cub::WarpReduce<float>;
    __shared__ typename WarpReduce::TempStorage temp;
    sum = WarpReduce(temp).Sum(sum);

    
    switch (weight_quant_num){
        case 0:
            if (threadIdx.x == 0)
                output[index] = sum;
            break;
        case 1:
            if (threadIdx.x == 0)
                output[index] = (half)sum; 
            break;
        default: 
            throw std::invalid_argument("Unknown weight quantization number");

    }

 
 }