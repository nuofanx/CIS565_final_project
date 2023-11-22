#pragma once 
#include <stdio.h>
#include <cuda_runtime.h>

// matrices are stored in row major order 
// M(row, col) = *(M.elements) + row * M.width + col
typedef struct{
    int width;
    int height;
    float* elements;
    int stride;
} Matrix;

#define BLOCK_SIZE = 16

__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);





