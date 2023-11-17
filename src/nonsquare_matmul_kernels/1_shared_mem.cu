#include "1_shared_mem.cuh"

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C){
    // block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

    // each thread compute one element of Csub by accumulating results into Cval
    float Cval = 0.0;

    // Thread row and col within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // loop over all submatrices of A and B
    for (int m =0; m<(A.width / BLOCK_SIZE); ++m){
        // get submatrix Asub of A
        Matrix Asub = GetSubMatrix(A, blockRow, m);
        Matrix Bsub = GetSubMatrix(B, m, blockCol);
        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // load Asub and Bsub from device memory to shared memory 
        // each thread loads one element of each submatrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);

        // sync to make sure the sub matrices are loaded properly
        __syncthreads();

        for (int e = 0; e < BLOCK_SIZE; ++e){
            Cval += As[row][e] * Bs[e][col];
        }
        // make sure preceding calcs are done before loading next two
        __syncthreads();

    }

    // write Csub to device memory. each thread writes one element 
    SetElement(Csub, row, col, Cval);
}
