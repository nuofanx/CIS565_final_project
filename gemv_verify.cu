#include <cuda_runtime.h>
#include <Windows.h>
#include <stdio.h>
#include "vector_add.cuh"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <cublas_v2.h>
#include <cub/cub.cuh>

#define BLOCK_HEIGHT 1024
#define BLOCK_WIDTH 64


int ceil_div(int a, int size){
    return (a -1) / size +1; 
}

#define _CUBLAS(x) do { if((x) != CUBLAS_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(EXIT_FAILURE);}} while(0)

void print_matrix(const float *A, int M, int N, std::ofstream &fs, std::string matrix_name) {
    fs << matrix_name << "\n";
    int i;
    fs << std::setprecision(2)
        << std::fixed; // Set floating-point precision and fixed notation
    fs << "[";
    for (i = 0; i < M * N; i++) {
        if ((i + 1) % N == 0) // if last element of the row 
        fs << std::setw(5) << A[i]; // Set field width and write the value
        else
        fs << std::setw(5) << A[i] << ", ";
        if ((i + 1) % N == 0) {
        if (i + 1 < M * N)
            fs << ";\n";
        }
    }
    fs << "]\n";
    fs << "\n";
}

__global__ void zero_vector_float(float *vec, const int n)
{
    unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if ( xIndex < n )
        vec[xIndex]=0.0f;
}

__global__ void MatMulKernel(float *out, float *in, float *a, const int matrixHeight, const int matrixWidth) {
    // get variables for loop
    // copy section of b into shared mem
    // go through the threads vertically and sum them into a variable
    // atomic add these variables to the corresponding c index

    // looping is happening horizontally on the matrix
    // BLOCK_WIDTH is again horizontal
    // BLOCK_HEIGHT is going vertical
    // n / BLOCK_WIDTH blocks horizontally
    // m / BLOCK_HEIGHT block vertically

    // get variables for loop
    // variable for loop length: blockEltHeight
    __shared__ int blockElt;
    __shared__ int blockxInd;
    __shared__ int blockyInd;
    if (threadIdx.x == 0) {
        if ((blockIdx.x + 1) * BLOCK_WIDTH <= matrixWidth)
            blockElt = BLOCK_WIDTH;
        else blockElt = matrixWidth % BLOCK_WIDTH;
        blockxInd = blockIdx.x * BLOCK_WIDTH;
        blockyInd = blockIdx.y * BLOCK_HEIGHT;
    }

    __syncthreads();
    
    // copy section of b into shared mem
    // use the first BLOCK_WIDTH of thread
    __shared__ float b[BLOCK_WIDTH];

    if (threadIdx.x < blockElt) 
        b[threadIdx.x] = in[blockxInd + threadIdx.x];
    
    __syncthreads();

    // summing variable
    float cSum = (float) 0;
    int threadyInd = blockyInd + threadIdx.x;

    // make sure we are inside the matrix verticallly
    if (threadyInd < matrixHeight) {
    
        // go through the threads vertically and sum them into a variable
        for (int i=0; i<blockElt; i++)
          // A col index   : blockIdx.x * BLOCK_WIDTH + i : blockxInd + i
          // A row index  : blockIdx.y * BLOCK_HEIGHT + threadIdx.x : blockyInd + threadIdx.x : threadyInd
          // B index : b[i]

          // cSum = B index * ( A col index * matrixHeight + A row index)
          cSum += b[i] * a[(blockxInd + i) * (matrixHeight) + (threadyInd)];
          //printf("csum = %f\n", cSum);
        
        // atomic add these variables to the corresponding c index
        atomicAdd(out + threadyInd, cSum);
    }
}

__global__ void MatMulKernelT(float *out, float *in, float *a, const int matrixHeight, const int matrixWidth) {
    // get variables for loop
    // copy section of b into shared mem
    // go through the threads vertically and sum them into a variable
    // atomic add these variables to the corresponding c index

    // looping is happening vertically on the matrix
    // BLOCK_WIDTH is going vertical
    // BLOCK_HEIGHT is going horizontal
    // m / BLOCK_WIDTH blocks vertically
    // n / BLOCK_HEIGHT block horizontally
  
    // get variables for loop
    // variable for loop length: blockElt
    __shared__ int blockElt;
    __shared__ int blockxInd;
    __shared__ int blockyInd;
    if (threadIdx.x == 0) {
        if ((blockIdx.y + 1) * BLOCK_WIDTH <= matrixHeight)
            blockElt = BLOCK_WIDTH;
        else blockElt = matrixHeight % BLOCK_WIDTH;
        blockxInd = blockIdx.x * BLOCK_HEIGHT;
        blockyInd = blockIdx.y * BLOCK_WIDTH;
    }
    
    __syncthreads();
    
    // copy section of b into shared mem
    // use the first BLOCK_WIDTH of thread
    __shared__ float b[BLOCK_WIDTH];

    if (threadIdx.x < blockElt)
        b[threadIdx.x] = in[blockyInd + threadIdx.x];
    
    __syncthreads();

    // summing variable
    float cSum = (float) 0;
    int threadxInd = blockxInd + threadIdx.x;

    // make sure we are inside the array horizontally
    if (threadxInd < matrixWidth) {
    
        // go through the threads vertically and sum them into a variable
        for (int i=0; i<blockElt; i++)
            // A col index : blockIdx.x * BLOCK_HEIGHT + threadIdx.x : blockxInd + threadIdx.x : threadxInd 
            // A row index : blockIdx.y * BLOCK_WIDTH + i : blockyInd + i
            // B index : b[i]

            // cSum = B index * ( A col index * matrixHeight + A row index)
            cSum += b[i] * a[(threadxInd) * (matrixHeight) + (blockyInd + i)];

        // atomic add these variables to the corresponding c index
        atomicAdd(out + threadxInd , cSum);
        //printf("el[%d%d;%d] csum = %f tot = %f\n", blockIdx.x, blockIdx.y, threadIdx.x, cSum, *(out + blockIdx.x * BLOCK_HEIGHT + threadIdx.x));
    }
}

float matVecMul (float * out, float * in, float * A, const int m, const int n)
{
    // in: vector (n, 1) 
    // A: matrix (m,n)
    // out: vector (m, 1)
    // set up threading and blocking variables
    cudaDeviceProp dp;
    cudaGetDeviceProperties(&dp,0);
    unsigned int max_threads_per_block = dp.maxThreadsPerBlock;

    int threads_perblockm = min(m, max_threads_per_block);
    dim3 threadsPerBlockm(threads_perblockm);
    int num_blocksm = (int)ceil((float)m/(float)threads_perblockm);
    dim3 numBlocksm(num_blocksm);

    int blockCols = (int) ceil(n / (double) BLOCK_WIDTH);
    int blockRows = (int) ceil(m / (double) BLOCK_HEIGHT);
    dim3 dimBlock(BLOCK_HEIGHT);
    dim3 dimGrid(blockCols, blockRows);

    int sharedMem = 3 * sizeof (int) + BLOCK_WIDTH * sizeof (float);

    // set up timing
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);

    // execute kernels
    zero_vector_float<<<numBlocksm, threadsPerBlockm>>>(out, m);
    MatMulKernel<<<dimGrid, dimBlock, sharedMem>>>(out, in, A, m, n);

    cudaThreadSynchronize();
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return time;
}

float matVecMulT (float * out, float * in, float * A, const int m, const int n)
{
    // set up threading and blocking variables
    cudaDeviceProp dp;
    cudaGetDeviceProperties(&dp,0);
    unsigned int max_threads_per_block = dp.maxThreadsPerBlock;

    int threads_perblockn = min(n, max_threads_per_block);
    dim3 threadsPerBlockn(threads_perblockn);
    int num_blocksn = (int)ceil((float)n/(float)threads_perblockn);
    dim3 numBlocksn(num_blocksn);

    int blockCols = (int) ceil(n / (double) BLOCK_HEIGHT);
    int blockRows = (int) ceil(m / (double) BLOCK_WIDTH);
    dim3 dimBlock(BLOCK_HEIGHT);
    dim3 dimGrid(blockCols, blockRows);

    int sharedMem = 3 * sizeof (int) + BLOCK_WIDTH * sizeof (float);

    // set up timing
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);

    // execute kernels
    zero_vector_float<<<numBlocksn, threadsPerBlockn>>>(out, n);
    MatMulKernelT<<<dimGrid, dimBlock, sharedMem>>>(out, in, A, m, n);

    cudaThreadSynchronize();
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return time;
}

template<const unsigned int BLOCK_DIM>
__global__ void transpose(float *odata, float *idata, int width, int height)
{
  __shared__ float block[BLOCK_DIM][BLOCK_DIM+1];
	
  // read the matrix tile into shared memory
  unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
  unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
  if((xIndex < width) && (yIndex < height))
    {
      unsigned int index_in = yIndex * width + xIndex;
      block[threadIdx.y][threadIdx.x] = idata[index_in];
    }

  __syncthreads();

  // write the transposed matrix tile to global memory
  xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
  yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
  if((xIndex < height) && (yIndex < width))
    {
      unsigned int index_out = yIndex * height + xIndex;
      odata[index_out] = block[threadIdx.x][threadIdx.y];
    }
}

//one output per warp so that we can parallelize the dot product across the warp
template<typename Type>
__global__ void matvecKernel_cubreduce(Type output, Type input, Type weight, const unsigned int input_dim, const unsigned int output_dim, const unsigned int hidden_dim) {
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

    if (threadIdx.x == 0){
        output[index] = sum;
    }
}
    

template <typename T, unsigned int BLOCK_SIZE> __global__ void matvec_kernel2(const T * __restrict__  dA, const T * __restrict__  dx, T * __restrict__ dy, const unsigned int nRows, const unsigned int nx){
    /**
     * Host-side wrapper for #matvec_kernel.
     *
     * @param   dA              Address of matrix `A` on the device
     * @param   dx              Address of vector `x` on the device
     * @param   dev_ptr_y       Address of result y = A*x
     * @param   nRows           Number of rows of `A`
     * @param   nx              Size of `x` (number of columns of `A`)
     * @param   elapsed_time    Time for the kernel to complete the execution in `ms`.
     *                          If NULL is passed to this argument, the elapsed time
     *                          will not be computed.
     *
     * @tparam  T               Data type for `A` and `x`
     */
       
        unsigned int bid = blockIdx.x;
        unsigned int row = threadIdx.x;
        const unsigned int block_size = blockDim.x;   // (nRows, nx) (nx,1)
        const unsigned int num_hor_blocks = ((nx + block_size - 1)/ block_size);
        unsigned int n_star;
        unsigned int idx_x;
        unsigned int idx_Asub;
        unsigned int idx_y;
        const T * Asub;
        const T * xsub;

        /* Only `x` is copied to shared memory */
        __shared__ T x_shared[BLOCK_SIZE];

        idx_y = bid * block_size;

        T * y_sub = dy + idx_y;

        T y_val = 0.0;

        #pragma unroll
        for (unsigned int m = 0; m < num_hor_blocks; ++m)
        {
            idx_Asub = block_size * (bid + m * nRows);
            idx_x = m * block_size;

            Asub = dA + idx_Asub;
            xsub = dx + idx_x;

            if (idx_x + row <  nx) {
                x_shared[row] = xsub[row];
            }

            __syncthreads();


            /* If the tiling is exact */
            if ( (nRows % block_size == 0 && nx % block_size == 0 ) ||
                    (m != block_size - 1 || bid != gridDim.x - 1)) {
                #pragma unroll
                for (int i = 0; i<BLOCK_SIZE;i++){
                    y_val += Asub[row+i*nRows] * x_shared[i];
                }
            } else {
                n_star = min(BLOCK_SIZE, nx - idx_x);
                #pragma unroll
                for (unsigned int e = 0; e < n_star; ++e) {
                    y_val += Asub[row + e * nRows] * x_shared[e];
                }
            }
            __syncthreads();
        }

        if (row + idx_y < nRows)
            y_sub[row] = y_val;

}

template<typename T, const unsigned int BLOCK_SIZE>
__global__ void matvec_kernel(const T * __restrict__ dA, const T * __restrict__ dx, T * __restrict__ dy, const unsigned int nRows, const unsigned int nCols)
{
    /**
     * Host-side wrapper for #matvec_kernel.
     *
     * @param   dA              Address of matrix `A` on the device
     * @param   dx              Address of vector `x` on the device
     * @param   dev_ptr_y       Address of result y = A*x
     * @param   nRows           Number of rows of `A`
     * @param   nx              Size of `x` (number of columns of `A`)
     * @param   elapsed_time    Time for the kernel to complete the execution in `ms`.
     *                          If NULL is passed to this argument, the elapsed time
     *                          will not be computed.
     *
     * @tparam  T               Data type for `A` and `x`
     */
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int num_horizontal_blks = (nCols + BLOCK_SIZE - 1)/ BLOCK_SIZE;
    __shared__ T x_shared[BLOCK_SIZE];
    T y_val = 0.0;
    #pragma unroll
    // loop through the blocks in horizontal direction 
    for (unsigned int m = 0; m < num_horizontal_blks; ++m)
    {
        // if thread not out of bound, copy to shared mem
        if ((m * BLOCK_SIZE + threadIdx.x) <  nCols) {
            x_shared[threadIdx.x] = dx[threadIdx.x + m * BLOCK_SIZE];
        }
        // otherwise fill with 0
        else{
            x_shared[threadIdx.x] = 0.f;
        }
        __syncthreads();
        #pragma unroll
        for (unsigned int e = 0; e < BLOCK_SIZE; ++e) {
            // --- Column-major ordering - faster
            y_val += dA[tid + (e + BLOCK_SIZE * m) * nRows] * x_shared[e];
            // --- Row-major ordering - slower
            //y_val += dA[tid * nCols + (e + BLOCK_SIZE * m)] * x_shared[e];
        }
        __syncthreads();
    }
    if (tid < nRows) dy[tid] = y_val;

}


void run_cublas_gemv(float * dx, float *dA, float * dev_ptr_y, int nRows, int nx, float alpha, float beta, bool column_major){
    //   run_cublas_gemv(d_a, d_b, d_cref, N, N, 1.0f, 0.0f);
      /**
     * Device-side wrapper for cublasSgemv_kernel that calculates dev_y = A*x of shape (nRows,1) = (nRows, nx) (nx, 1).
     *
     * @param   dA              Address of matrix `A` on the device
     * @param   dx              Address of vector `x` on the device
     * @param   dev_ptr_y       Address of result y = A*x
     * @param   nRows           Number of rows of `A`
     * @param   nx              Size of `x` (number of columns of `A`)        
     * @param   alpha           coefficient of gemv: y=alpha*A*x+beta*y, set to 1 to obtain y = A*x
     * @param   beta            coefficient of gemv: y=alpha*A*x+beta*y, set to 0 to obtain y = A*x
     *
     * @tparam  T               Data type for `A` and `x`
     */
    cublasStatus_t stat;   // cuBLAS functions status
    cublasHandle_t handle; // cuBLAS context
    stat = cublasCreate(&handle); // initialize CUBLAS context
    // row major order 
    if (!column_major)
            // To use two-dimensional arrays stored in row-major order in cublas (that works with column-major order) you can call the gemv in this way.
        _CUBLAS(cublasSgemv(handle, CUBLAS_OP_T, nx, nRows, &alpha, dA, nx, dx, 1, &beta, dev_ptr_y, 1));
    else
        // column major order matrices
        _CUBLAS(cublasSgemv(handle, CUBLAS_OP_N, nRows, nx, &alpha, dA, nRows, dx, 1, &beta, dev_ptr_y, 1));
    cublasDestroy(handle);
}
//   _CUBLAS(cublasSgemv(handle, CUBLAS_OP_N, nrows, ncols, &alpha, dev_rand_data + ncols,
                                    // nrows, dev_rand_data, 1, &beta, dev_y_cublas, 1));
// cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F,
            //    N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
            //    CUBLAS_GEMM_DEFAULT_TENSOR_OP);

int main(){
    std::ofstream fs;
    const std::string LogFile = "mv_result.txt";
    fs.open(LogFile);

    float *a = nullptr, *b=nullptr, *out=nullptr, *out_ref = nullptr;  // host side pointers
    float *d_a;      // device side vector pointer
    float *d_b;      // device side matrix pointer
    float *d_c;      // device side output pointer for custom kernel  
    float *d_cref;   // device side output pointer for cublas  
    const unsigned int N =3;
    a = (float*)malloc(sizeof(float) * N);     // vector of shape (N, 1)
    b = (float*)malloc(sizeof(float) * N*N);   // matrix of shape (N, N)
    out=(float*)malloc(sizeof(float) * N);     // for custom kernel output 
    out_ref=(float*)malloc(sizeof(float) * N); // for cublas calculation output 

    printf("filling host arrays\n");
    Sleep(1000);
    // Fill host arrays A and B
    for(int i=0; i<N; i++)
    {
        a[i] = 1.0f/(i+1);    
    }
    for (int i=0; i<N*N;i++){
        b[i] = 2.0f/(i+1);
    }
    print_matrix(a, N, 1, fs, "vector x");
    print_matrix(b, N, N, fs, "matrix A");
    printf("filling host arrays done\n");
    Sleep(1000);

    // Allocate device memory for a
    cudaMalloc((void**)&d_a, sizeof(float) * N);
    cudaMalloc((void**)&d_b, sizeof(float) * N*N);
    cudaMalloc((void**)&d_c, sizeof(float) * N);
    cudaMalloc((void**)&d_cref, sizeof(float) * N);
    // Transfer data from host to device memory
    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N*N, cudaMemcpyHostToDevice);
    printf("calling mat vec multiplication kernel\n");
    // blockNum, threadNum
    // vector_add<<<1,1>>>(out, d_a, d_b, N);
   
    // 32 threads per warp, use block_size of 32 to break vector a into chunks
    // use 32 threads x 4 threads per block
    // each thread processes 
    // dim3 block_dim(32,4);
    // int blocks = ceil_div(N,4);
    // bool is_column_major = false; // if (!is_column_major) run_cublas_gemv(row)
    // matvecKernel_cubreduce<<<blocks, block_dim>>>(d_c, d_a, d_b, N, N, ceil_div(N,32));
   
    // int thr_per_blk = 256;
    // int blk_in_grid = ceil( float(N) / thr_per_blk );
    // matvec_kernel2<float, 16><<<blk_in_grid, thr_per_blk>>>(d_b, d_a, d_c, N, N);
    // matvec_kernel<float,16><<<blk_in_grid, thr_per_blk>>>(d_b, d_a, d_c, N, N);
    bool is_column_major = true; // if (!is_column_major) run_cublas_gemv(row)
    matVecMul(d_c, d_a, d_b, N, N); // y=A^Tx, d_b is matrix (m, n), d_a is (n, 1)
    run_cublas_gemv(d_a, d_b, d_cref, N, N, 1.0f, 0.0f, is_column_major); // if is_column_major, use 
    printf("kernel call done\n");
    cudaMemcpy(out, d_c, sizeof(float) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(out_ref, d_cref, sizeof(float) * N, cudaMemcpyDeviceToHost);
    // Cleanup after kernel execution
    
    printf("writing to text file\n");
    print_matrix(out, N, 1, fs, "custom kernel result");
    print_matrix(out_ref, N, 1, fs, "cublas result");
    printf("writing done");
    cudaFree(d_a);
    cudaFree(d_b);
    free(a);
    free(b);
    fs.close();
    Sleep(2000);
}