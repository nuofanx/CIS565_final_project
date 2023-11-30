#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
// cuda specific headers
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cub/cub.cuh>
// Navidia cublas lib for matmul and matvec mul
#include <cublas_v2.h>

#if defined _WIN32
    #include <Windows.h>
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif

#define writeToFile(...) fprintf(File, __VA_ARGS__)

#define MAX_SEQ_LEN_SMEM_KERNEL 8192 // user defined max sequence length supported by the kernel that uses shared memory
#define MAX_SEQ_LEN 8192 // user defined the max sequence length 

// utility kernels
int divUp(int a, int b) {
    /**
     * Function to perform ceil(a,b)
     *
     * @param   a         Numerator
     * @param   b         Denominator 
     */

    return (a - 1) / b + 1;
}

template<typename T>
void alloc_gpu_space(T* w, int n, size_t size){
    cudaMalloc((void**)&w,  n * size);
}


template<typename T1>
void malloc_space(T1* x, T1* q, T1* k, T1* v, T1* att, int n_layers, int n_heads, int seq_len, int dim, int n_kv_heads) {
    /**
     * Function to allocate device memory
     *
     * @param   s          Address of RunState struct 
     * @param   p          Address of Config struct 
     *
     * @tparam  T1         Data type of device params
     * @tparam  T2         Data type of host params
     */
    int kv_dim = (dim * n_kv_heads) / n_heads;
    alloc_gpu_space(x, dim, sizeof(T1));
    alloc_gpu_space(q, dim, sizeof(T1));
    alloc_gpu_space(att, n_heads * dim, sizeof(T1));
    alloc_gpu_space(k, n_layers * seq_len * kv_dim, sizeof(T1));    // potentially huge allocs
    alloc_gpu_space(v, n_layers * seq_len * kv_dim, sizeof(T1));
   
}

template<typename T1>
void malloc_space_nonfusion(T1* x, T1* q, T1* k, T1* v, int n_layers, int n_heads, int seq_len, int dim) {
    alloc_gpu_space(x, dim, sizeof(T1));
    alloc_gpu_space(q, dim, sizeof(T1));
    alloc_gpu_space(k, n_layers * seq_len * dim, sizeof(T1));    // potentially huge allocs
    alloc_gpu_space(v, n_layers * seq_len * dim, sizeof(T1));
}

template<typename T1>
void free_space(T1* x, T1* q, T1* k, T1* v, T1* att) {
    /**
     * Function to free device memory for RunState struct 
     *
     * @param   s          Address of RunState struct  
     *
     * @tparam  T1         Data type of device params
     * @tparam  T2         Data type of host params
     */

    cudaFree(x);
    cudaFree(q);
    cudaFree(k);
    cudaFree(v);
    cudaFree(att);
}

template<typename T1>
void free_space(T1* x, T1* q, T1* k, T1* v) {
    /**
     * Function to free device memory for RunState struct 
     *
     * @param   s          Address of RunState struct  
     *
     * @tparam  T1         Data type of device params
     * @tparam  T2         Data type of host params
     */

    cudaFree(x);
    cudaFree(q);
    cudaFree(k);
    cudaFree(v);
}

// fusion kernels 
template <typename T>
__global__ void mat_vec_kernel_MHA(T* op, const T* ip, const T* wt, int n, int numSerialElements,
    int ip_stride, int w_stride, int w_row_stride, float alpha, int pPos, int kv_mul) {

    int op_stride = pPos + 1;
    int index = blockIdx.x * blockDim.y + threadIdx.y;
    if (index >= op_stride)
        return;

    const T* __restrict__ input = ip + blockIdx.y * ip_stride;
    const T* __restrict__ weight = wt + (blockIdx.y / kv_mul) * w_stride;
    T* output = op + blockIdx.y * op_stride;

    float sum = 0;
    for (int i = 0; i < numSerialElements; i++) {
        int j = i * 32 + threadIdx.x;
        if (j < n)
            sum += ((float)weight[index * w_row_stride + j]) * ((float)input[j]);
    }

    using WarpReduce = cub::WarpReduce<float>;
    __shared__ typename WarpReduce::TempStorage temp;
    sum = WarpReduce(temp).Sum(sum);
    sum *= alpha;

    if (threadIdx.x == 0)
        output[index] = (T)sum;

}


template <typename T>
__global__ void softmax_kernel(T* __restrict__ arr, int num_heads, int pos) {
    __shared__ float att[MAX_SEQ_LEN_SMEM_KERNEL];
    int h = blockIdx.x;
    int tid = threadIdx.x;
    int step = blockDim.x;
    int size = pos + 1;

    // load input to shared memory
    for (int t = tid; t < size; t += step)
        att[t] = (float)arr[h * size + t];
    __syncthreads();

    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_val;

    // find max value (for numerical stability)
    float max_val = tid < size ? att[tid] : 0;
    for (int i = tid + step; i < size; i += step)
        if (att[i] > max_val)
            max_val = att[i];

    max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
    if (threadIdx.x == 0)
        shared_val = max_val;
    __syncthreads();
    max_val = shared_val;

    // exp and sum
    float sum = 0.0f;
    for (int i = tid; i < size; i += step) {
        att[i] = expf(att[i] - max_val);
        sum += att[i];
    }

    sum = BlockReduce(temp).Sum(sum);
    if (threadIdx.x == 0)
        shared_val = sum;
    __syncthreads();
    sum = shared_val;

    // normalize and write the result
    for (int t = tid; t < size; t += step)
        arr[h * size + t] = (T)(att[t] / sum);
}


template <typename T>
__global__ void vec_mat_kernel_MHA(T* op, T* __restrict__ ip, T* __restrict__ wt,
                               int N, int* pPos, int w_stride, int op_stride, int w_row_stride, int kv_mul) {
    int K = *pPos + 1;

    // Compute pointers to input, weight, and output
    const T* __restrict__ input = ip + blockIdx.y * K;
    const T* __restrict__ weight = wt + (blockIdx.y / kv_mul) * w_stride;
    T* output = op + blockIdx.y * op_stride;

    int start_n = blockIdx.x * 32;
    int i = start_n + threadIdx.y;

    // Double buffering for shared memory
    __shared__ T loaded_fragment[2][32][32 + 2];

    // Check for out-of-bounds threads
    if (i >= N)
        return;

    // Load the first 32x32 fragment into shared memory
    int n = start_n + threadIdx.x;
    int k = threadIdx.y;
    int offset = k * w_row_stride + n;
    loaded_fragment[0][threadIdx.y][threadIdx.x] = ((n < N) && (k < K)) ? weight[offset] : (half)0.0;

    float sum = 0;

    // Loop over the matrix row and vector elements
    for (int e = 0; ;) {
        __syncthreads(); // Synchronize threads to wait for the load

        int start_k = e * 32;
        if (start_k >= K) break;
        k = start_k + threadIdx.x;
        int buf_i = e & 1;
        sum += float(loaded_fragment[buf_i][threadIdx.x][threadIdx.y]) * ((k < K) ? (float)input[k] : 0.0f);

        // Load for the next iteration
        e++;
        start_k = e * 32;
        buf_i = e & 1;
        n = start_n + threadIdx.x;
        k = start_k + threadIdx.y;
        offset = k * w_row_stride + n;
        loaded_fragment[buf_i][threadIdx.y][threadIdx.x] = ((n < N) && (k < K)) ? weight[offset] : (half)0.0;
    }

    // Reduce sum within warp using WarpReduce from cub library
    using WarpReduce = cub::WarpReduce<float>;
    __shared__ typename WarpReduce::TempStorage temp;
    sum = WarpReduce(temp).Sum(sum);

    // Write back the result to output
    if (threadIdx.x == 0)
        output[i] = (T)sum;
}

template <typename T>
__global__ void softmax_kernel_no_smem(T* arr, int num_heads, int pos) {
    int h = blockIdx.x;
    int tid = threadIdx.x;
    int step = blockDim.x;
    int size = pos + 1;

    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_val;

    // find max value (for numerical stability)
    float max_val = tid < size ? (float)arr[h * size + tid] : 0;
    for (int i = tid + step; i < size; i += step)
    {
        float val = (float)arr[h * size + i];
        if (val > max_val)
            max_val = val;
    }

    max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
    if (threadIdx.x == 0)
        shared_val = max_val;
    __syncthreads();
    max_val = shared_val;

    // exp and sum
    float sum = 0.0f;
    for (int i = tid; i < size; i += step) {
        float val = (float)arr[h * size + i];
        val = expf(val - max_val);
        arr[h * size + i] = (T)val;
        sum += val;
    }

    sum = BlockReduce(temp).Sum(sum);
    if (threadIdx.x == 0)
        shared_val = sum;
    __syncthreads();
    sum = shared_val;

    // normalize and write the result
    for (int t = tid; t < size; t += step)
        arr[h * size + t] = (T)(float(arr[h * size + t]) / sum);
}


__device__ void softmax_gpu(float* __restrict__ x, int size) {
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



template <typename T>
void MultiHeadAttentionFused(T *output, T *q, T *key_cache, T* value_cache, T *att, int num_heads, int head_size, int kv_mul, int max_seq_len, int* ppos) {
    int dim = head_size * num_heads;
    int serialElements = divUp(head_size, 32);
    dim3 block_dim(32, 32);
    dim3 grid_dim1(divUp(max_seq_len, 32), num_heads);      // using max_seq_len instead of real seq_len here has measurable impact on perf (2%) :-/
    mat_vec_kernel_MHA <<< grid_dim1, block_dim>>> (att, q, key_cache, head_size, serialElements, head_size, head_size, dim / kv_mul, 1.0 / sqrt(head_size), *ppos, kv_mul);
    
    // 2. Run softmax kernel
    if (max_seq_len <= MAX_SEQ_LEN_SMEM_KERNEL)
        softmax_kernel <<< num_heads, 1024>>> (att, num_heads, *ppos);
    else
        softmax_kernel_no_smem <<< num_heads, 1024>>> (att, num_heads, *ppos);

    // 3. weighted sum of the values to get the final result
    dim3 grid_dim2(divUp(head_size, 32), num_heads);
    vec_mat_kernel_MHA <<< grid_dim2, block_dim>>> (output, att, value_cache, head_size, ppos, head_size, head_size, dim / kv_mul, kv_mul);
}

//each block processes a single head
template <typename T>
__global__ void MultiHeadAttention_kernel(T* __restrict__ output, const T* __restrict__ sq,
    const T* __restrict__ key_cache, const T* __restrict__ value_cache,
    int num_heads, int head_size, int loff, int seq_len, int dim) {
    /**
     * Kernel function to run SiLU element wise multiplication  
     * @param   output            Address of output vector 
     * @param   sq                Address of Q vector 
     * @param   key_cache         Address of K vector 
     * @param   value_cache       Address of V vector
     * @param   num_heads         Number of attention heads
     * @param   head_size         Number of attention heads 
     * @param   loff              load offset 
     * @param   seq_len           maximum length of the sequence 
     * 
     * @tparam  T                 Data type of device params
     */
    
    int h = blockIdx.x;

    // get the query vector for this head
    const T* q = sq + h * head_size;
    // attention scores for this head
    __shared__ float att[MAX_SEQ_LEN];

    // iterate over all timesteps, including the current one
    for (int t = threadIdx.x; t < seq_len; t += blockDim.x) {
        // get the key vector for this head and at this timestep
        const T* k = key_cache + loff + t * dim + h * head_size;
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

    // weighted sum of the values, store back into xb
    for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
        float val = 0.0f;
        for (int t = 0; t < seq_len; t++)
            val += att[t] * (float)value_cache[loff + t * dim + h * head_size + i];
        output[h * head_size + i] = (T)val;
    }
}

template <typename T>
void MultiHeadAttention(T *output, T *q, T *key_cache, T *value_cache, int num_heads, int head_size, int loff, int seq_len) {
    /**
     * Wrapper Function to run RoPERotation_kernel
     *
     * @param   output            Address of output vector 
     * @param   q                 Address of Q vector 
     * @param   key_cache         Address of K vector 
     * @param   value_cache       Address of V vector
     * @param   num_heads         Number of attention heads
     * @param   head_size         Number of attention heads 
     * @param   loff              load offset 
     * @param   seq_len           maximum length of the sequence 
     * 
     * @tparam  T                 Data type of device params
     */

    int dim = head_size * num_heads;
    MultiHeadAttention_kernel <<<num_heads, 1024>>> (output, q, key_cache, value_cache, num_heads, head_size, loff, seq_len, dim);
}

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    timespec_get(&time, TIME_UTC);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

void record(FILE * File, int n_heads, int n_kv_heads, int seq_len, int dim, int hidden_dim, int n_layers){
    int kv_mul = n_heads / n_kv_heads;
    int head_size = dim / n_heads;
    int pos = 0;
    long start = time_in_ms(); 
    float *x; float* att; float* q; float* k; float* v;
    malloc_space(x, q, k, v, att, n_layers, n_heads, seq_len, dim, n_kv_heads);
    int repeated_times = 5000;
    for (int i =0; i<repeated_times; i++){
        MultiHeadAttentionFused(x, q, k, v, att, n_heads, head_size, kv_mul, seq_len, &pos);
    }
    long end = time_in_ms();
    float time1 = (end - start) / 1000.0;
    float *x2; float* q2; float* k2; float* v2;
    malloc_space_nonfusion(x2, q2, k2, v2, n_layers, n_heads, seq_len, dim);
    long start2 = time_in_ms();
    for (int i =0; i<repeated_times; i++){
        MultiHeadAttention(x, q2, k2, v2, n_heads, head_size, 0, seq_len);
    }
    long end2 = time_in_ms();
    float time2 = (end2 - start2) / 1000.0;
    printf("size:(%d), time1: (%f), time2: (%f)\n", dim, time1, time2);
    free_space(x,q,k,v,att);
    free_space(x2,q2,k2,v2);
    writeToFile("dim: (%d), time1: (%f), time2: %f(\n)", dim, time1, time2);

}

int main(){
    FILE * File;
    std::ostringstream stringStream;    
    stringStream << "benchmark_results/mha/output.txt";
    std::string filename = stringStream.str();

    File = fopen(filename.c_str(), "w+");
    int n_heads = 6;
    int n_kv_heads= 6;
    int seq_len = 256;
    int pos = 0;
    int dim = 288;
    int hidden_dim = 2048;
    int n_layers = 6;
    record(File, n_heads, n_kv_heads, seq_len, dim, hidden_dim, n_layers);
    int n_heads2 = 32;
    int n_kv_heads2= 32;
    int seq_len2 = 256;
    int pos2 = 0;
    int dim2 = 1024;
    int hidden_dim2 = 2048;
    int n_layers2 = 32;
    record(File, n_heads2, n_kv_heads2, seq_len2, dim2, hidden_dim2, n_layers2);
    int n_heads3 = 16;
    int n_kv_heads3 = 16;
    int seq_len3 = 256;
    int pos3 = 0;
    int dim3 = 512;
    int hidden_dim3 = 2048;
    int n_layers3 = 16;
    record(File, n_heads3, n_kv_heads3, seq_len3, dim3, hidden_dim3, n_layers3);

}