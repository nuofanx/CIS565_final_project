#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>
#include <Windows.h>
#include <io.h>
#include <fcntl.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>
#include <iomanip>
#include <cub/cub.cuh>
#include <iostream>
 
#define writeToFile(...) fprintf(File, __VA_ARGS__)
#define BLOCK_HEIGHT 1024
#define BLOCK_WIDTH 64


#define _CUBLAS(x) do { if((x) != CUBLAS_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(EXIT_FAILURE);}} while(0)

int ceil_div(int a, int size){
    return (a -1) / size +1; 
}

void cudaCheckInfo(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
            cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

void cudaCheck(cudaError_t err){ cudaCheckInfo(err, __FILE__, __LINE__);}

void randomize_matrix(float *mat, int N) {
     /**
     * Function to init matrix/vector `mat` with randomized float
     *
     * @param   mat             Address of matrix/vector `mat` on the host 
     * @param   N               The total number of elements that need to be initialized
     *
     * @tparam  T               Data type for matrix/vector `mat`
     */

    // NOTICE: Use gettimeofday instead of srand((unsigned)time(NULL)); the time
    // precision is too low and the same random number is generated.
    
    srand((unsigned)time(NULL));
    for (int i = 0; i < N; i++) {
        float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
        tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
        mat[i] = tmp;
    }
}

bool verify_matrix(float *matRef, float *matOut, int N) {
     /**
     * Function to verify matrix matRef is the same as matrix matOut
     *
     * @param   matRef          Address of matrix/vector matRef on the host 
     * @param   matOut          Address of matrix/vector matOut on the host
     * @param   N               The total number of elements that need to be checked
     *
     * @tparam  T               Data type for matrix/vector `A`
     */

    double diff = 0.0;
    int i;
    for (i = 0; i < N; i++) {
        diff = std::fabs(matRef[i] - matOut[i]);
        if (diff > 0.01) {
        printf("Divergence! Should %5.2f, Is %5.2f (Diff %5.2f) at %d\n",
                matRef[i], matOut[i], diff, i);
        return false;
        }
    }
    return true;
}

void print_matrix(const float *A, int M, int N, std::ofstream &fs, std::string matrix_name) {
     /**
     * Function to print matrix/vector to ofstream fs with string matrix name 
     *
     * @param   A               Address of matrix/vector `A` on the host 
     * @param   M               First dimension of matrix/vector A 
     * @param   N               Second dimension of matrix/vector A, 1 if vector
     * @param   fs              ofstream 
     * @param   matrix_name     The custom name of matrix/vector A 
     *
     * @tparam  T               Data type for matrix/vector `A`
     */

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

//one output per warp so that we can parallelize the dot product across the warp
template<typename T>
__global__ void matvecKernel_cubreduce(T output, T input, T weight, const unsigned int input_dim, const unsigned int output_dim, const unsigned int hidden_dim) {
    /**
     *  Kernel function that calculates output = weight*input of shape (nRows,1) = (nRows, nx) (nx, 1).
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


void run_cublas_gemv(float * dx, float *dA, float * dev_ptr_y, int nRows, int nx, float alpha, float beta, bool column_major){
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

void run_kernel(int kernel_num, float*dA, float*dB, float*dC, int N){
    if (kernel_num == 0) // row major order
        run_cublas_gemv(dA, dB, dC, N, N, 1.0f, 0.0f, false); 
    if (kernel_num == 1){ // row major order
        int blocks = ceil_div(N,4);
        dim3 block_dim(32,4);
        matvecKernel_cubreduce<<<blocks, block_dim>>>(dC, dA, dB, N, N, ceil_div(N,32));
    }
    if (kernel_num == 2) //column major order
        run_cublas_gemv(dA, dB, dC, N, N, 1.0f, 0.0f, true);
    if (kernel_num == 3) //column major order
        matVecMul(dC, dA, dB, N, N);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Please select a kernel (range 0 - 3, 0 and 2 for NVIDIA cuBLAS row major and column major respectively)"
                << std::endl;
        Sleep(5000);
        exit(EXIT_FAILURE);
    }

    // get kernel number
    int kernel_num = std::atoi(argv[1]);
    if (kernel_num < 0 || kernel_num > 12) {
        std::cerr << "Please enter a valid kernel number (0-3)" << std::endl;
        Sleep(5000);
        exit(EXIT_FAILURE);
    }

    // get environment variable for device
    int deviceIdx = 0;
    if (getenv("DEVICE") != NULL) {
        deviceIdx = atoi(getenv("DEVICE"));
    }
    cudaCheck(cudaSetDevice(deviceIdx));

    printf("Running kernel %d on device %d.\n", kernel_num, deviceIdx);
    Sleep(1000);
    // print some device info
    // CudaDeviceInfo();

    // Declare the handle, create the handle, cublasCreate will return a value of
    // type cublasStatus_t to determine whether the handle was created
    // successfully (the value is 0)
    cublasHandle_t handle;
    if (cublasCreate(&handle)) {
        std::cerr << "Create cublas handle error." << std::endl;
        Sleep(1000);
        exit(EXIT_FAILURE);
    };

    // Using cudaEvent for gpu stream timing, cudaEvent is equivalent to
    // publishing event tasks in the target stream
    printf("start timing\n");
    float elapsed_time;
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);

    // cuBLAS FLOPs ceiling is reached at 8192
    std::vector<int> SIZE = {64, 128, 256, 512, 1024, 2048, 4096};
    // std::vector<int> SIZE = {2, 4};

    long nRows, nx, max_size;
    max_size = SIZE[SIZE.size() - 1];
    std::cout << "Max size: " << max_size << std::endl;

    float alpha = 1.0f, beta = 0.0f; // GEMV input parameters, y=alpha*Ax+beta*y

    float *A = nullptr, *B = nullptr, *C = nullptr,
            *C_ref = nullptr; // host matrices
    float *dA = nullptr, *dB = nullptr, *dC = nullptr,
            *dC_ref = nullptr; // device matrices

    A = (float *)malloc(sizeof(float) * max_size); // vec
    B = (float *)malloc(sizeof(float) * max_size * max_size); // mat
    C = (float *)malloc(sizeof(float) * max_size); // vec 
    C_ref = (float *)malloc(sizeof(float) * max_size); // vec

    randomize_matrix(A, max_size);
    randomize_matrix(B, max_size * max_size);
    std::ofstream fs;
    const std::string LogFile = "gemv_input.txt";
    fs.open(LogFile);
    printf("opening file\n");
    if (max_size < 5){
        print_matrix(A, max_size, 1, fs, "vector");
        print_matrix(B, max_size, max_size, fs, "matrix");
    }
    
    Sleep(1000);

    printf("init cuda memory\n");
    cudaCheck(cudaMalloc((void **)&dA, sizeof(float) * max_size));
    cudaCheck(cudaMalloc((void **)&dB, sizeof(float) * max_size * max_size));
    cudaCheck(cudaMalloc((void **)&dC, sizeof(float) * max_size));
    cudaCheck(cudaMalloc((void **)&dC_ref, sizeof(float) * max_size));

    cudaCheck(cudaMemcpy(dA, A, sizeof(float) * max_size,
                        cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dB, B, sizeof(float) * max_size * max_size,
                        cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC, C, sizeof(float) * max_size,
                        cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC_ref, C, sizeof(float) * max_size,
                        cudaMemcpyHostToDevice));
   
    int thr_per_blk = 256;
    int repeat_times = 1;
    FILE * File;
    std::ostringstream stringStream;    
    stringStream << "benchmark_results/sgemv/" << kernel_num << "_output.txt";
    std::string filename = stringStream.str();

    File = fopen(filename.c_str(), "w+");
   

    for (int size : SIZE) {
        printf("current size %ld \n", size);
        nRows = nx = size;

        int N = size;    
        int blk_in_grid = ceil( float(N) / thr_per_blk );

        std::cout << "dimensions (nRows, nx)*(nx, 1), nRows:" << nRows << ", nx: " << nx << ", alpha: " << alpha
                << ", beta: " << beta << std::endl;
        Sleep(1000);
        // Verify the correctness of the calculation, and execute it once before the
        // kernel function timing to avoid cold start errors
        printf("running custom mat vec mul kernel %ld \n", kernel_num);
        Sleep(1000);
       
        run_kernel(kernel_num, dA, dB, dC, N);
        printf("running cublas kernel \n");
        Sleep(1000);
        bool is_column_major;
        if (kernel_num <= 1)
            is_column_major = false; // stored in row major natrually 
        else
            is_column_major = true;  // wrong calculation but pretend to be column major order 
        run_cublas_gemv(dA, dB, dC_ref, N, N, 1.0f, 0.0f, is_column_major);
        cudaCheck(cudaDeviceSynchronize());
        cudaCheck(cudaGetLastError()); // Check for async errors during kernel run
        printf("calculations done\n");
        Sleep(2000);
        cudaMemcpy(C, dC, sizeof(float) * nRows, cudaMemcpyDeviceToHost);
        cudaMemcpy(C_ref, dC_ref, sizeof(float) * nRows, cudaMemcpyDeviceToHost);
        if (max_size <5){
            fs << "current dimensions (nRows, nx)*(nx, 1), nRows:" << nRows << ", nx: " << nx << std::endl;
            print_matrix(C, nx, 1, fs, "custom kernel result");
            print_matrix(C_ref, nx, 1, fs, "Cublas result");
        }
 
        std::string errLogFile = "errLog.txt";
        printf("verify result \n");
        Sleep(1000);
        if (!verify_matrix(C_ref, C, nRows)) {
            Sleep(1000);
            std::cout
                << "Failed to pass the correctness verification against NVIDIA "
                "cuBLAS."
                << std::endl;
            std::cout << " Logging faulty output into " << errLogFile << "\n";
            std::ofstream fs_err;
            fs_err.open(errLogFile);
            fs_err << "A:\n";
            print_matrix(A, nx, 1, fs_err, "vector x");
            fs_err << "B:\n";
            print_matrix(B, nRows, nx, fs_err, "matrix A");
            fs_err << "C:\n";
            print_matrix(C, nx, 1, fs_err, "result y");
            fs_err << "Should:\n";
            print_matrix(C_ref, nx, 1, fs_err, "reference y_ref");
            fs_err.close();
            exit(EXIT_FAILURE);
        }
        printf("verified! \n");
        cudaEventRecord(beg);
        for (int j = 0; j < repeat_times; j++) {
            // We don't reset dC between runs to save time
            run_kernel(kernel_num, dA, dB, dC, N);
        }

        cudaEventRecord(end);
        cudaEventSynchronize(beg);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsed_time, beg, end);
        elapsed_time /= 1000.; // Convert to seconds

        long flops = 2 * nRows * nx * 1;
        writeToFile(
            "Average elapsed time: (%7.6f) s, performance: (%7.6f) MFLOPS. size: (%ld x %ld).\n",
            elapsed_time / repeat_times,
            (repeat_times * flops * 1e-6) / elapsed_time, nRows, nx);
        // make dC and dC_ref equal again (we modified dC while calling our kernel
        // for benchmarking)
        cudaCheck(cudaMemcpy(dC, dC_ref, sizeof(float) * nRows,
                            cudaMemcpyDeviceToDevice));
    }

    // Free up CPU and GPU space
    printf("ALL COMPLETE, free memory\n");

    free(A);
    free(B);
    free(C);
    free(C_ref);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(dC_ref);
    cublasDestroy(handle);
    fs.close();
    fclose(File);
    Sleep(2000);
    return 0;
};