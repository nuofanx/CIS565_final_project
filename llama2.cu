/*
Inference for Llama-2 Transformer model in pure Cuda.
*/

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

// ----------------------------------------------------------------------------
#define MAX_SEQ_LEN_SMEM_KERNEL 8192 // user defined max sequence length supported by the kernel that uses shared memory
#define MAX_SEQ_LEN 8192 // user defined the max sequence length 

#define writeToFile(...) fprintf(File, __VA_ARGS__)

// GPU kernels
template<typename T>
__global__ void element_wise_add_kernel(T* dest, T* src, int size) {
    /**
     * Kernel function to run atomic add on GPU
     *
     * @param   dest              Address of matrix/vector dest
     * @param   src               Address of matrix/vector src
     * @param   size              Number of elements 
     * 
     * @tparam  T                 Data type of device params
     */

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        dest[i] = (T)((float)dest[i] + (float)src[i]);
}

template <typename T1, typename T2>
__global__ void convert(T1* out, T2* in, int elements){
    /**
     * Convert in vector of type T2 to type T1 
     *
     * @param   out               Address of output vector 
     * @param   in                Address of in vector 
     * @param   elements          Number of elements in vector to convert 
     * 
     * @tparam  T1                Data type of device params
     * @tparam  T2                Data type of host params
     */
    out = (T1*) out;
    int index = blockIdx.x * 256 + threadIdx.x;
    if (index < elements)
        out[index] = (T1)in[index];
}

// Single block - not enough parallelism for the GPU, but it's just 1% of total time
template <typename T>
__global__ void rmsnorm_kernel(T* o, T* x, T* weight, int size, int elementsPerThread) {
    /**
     * Kernel function to run rmsnorm   
     *
     * @param   o                 Address of output vector 
     * @param   x                 Address of state vector 
     * @param   weight            Address of weight matrix
     * @param   size              Number of elements in state vector 
     * @param   elementsPerThread Number of elements per thread 
     * 
     * @tparam  T                 Data type of device params
     */
    float ss = 0.0f;
    for (int i = 0; i < elementsPerThread; i++) {
        int index = threadIdx.x + i * 1024;
        if (index < size)
            ss += (float)x[index];
    }

    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage temp;
    ss = BlockReduce(temp).Sum(ss * ss);

    __shared__ float shared_ss;
    if (threadIdx.x == 0) {
        ss /= size;
        ss += 1e-5f;
        ss = 1.0f / sqrtf(ss);
        shared_ss = ss;
    }
    __syncthreads();
    ss = shared_ss;

    // normalize
    for (int i = 0; i < elementsPerThread; i++) {
        int index = threadIdx.x + i * 1024;
        if (index < size) {
            float val = (float)x[index];
            val *= ss * (float)weight[index];
            o[index] = (T)val;
        }
    }
}

// one output per warp so that we can parallelize the dot product across the warp
// Note that ~95% of total time is spent here, so optimizing this is important
template <typename T>
__global__ void mat_vec_kernel(T* output, T* input, T* weight, int n, int d, int numSerialElements) {

    /**
     * Wrapper Function to run mat vec multiplication kernel  
     *
     * @param   output               Address of output vector 
     * @param   intput               Address of state vector 
     * @param   weight               Address of weight matrix
     * @param   n                    Number of elements in state vector 
     * @param   d                    Number of rows of weight matrix 
     * @param   numSerialElements    Number of warps 
     * 
     * @tparam  T                    Data type of device params
     */

    int index = blockIdx.x * blockDim.y + threadIdx.y;
    if (index >= d)
        return;

    float sum = 0;
    for (int i = 0; i < numSerialElements; i++) {
        int j = i * 32 + threadIdx.x;
        if (j < n)
            sum += ((float)weight[index * n + j]) * ((float)input[j]);
    }

    using WarpReduce = cub::WarpReduce<float>;
    __shared__ typename WarpReduce::TempStorage temp;
    sum = WarpReduce(temp).Sum(sum);

    if (threadIdx.x == 0)
        output[index] = (T)sum;
}

// Each block processes a single head
template <typename T>
__global__ void RoPERotation_kernel(T* sq, T* sk, T* f_real, T* f_imag, int num_heads, int head_size) {
     /**
     * Wrapper Function to run RoPERotation_kernel
     *
     * @param   sq                Starting address of Q vector 
     * @param   sk                Starting address of K vector 
     * @param   f_real            Address of f_real
     * @param   f_img             Address of f_img  
     * @param   num_heads         Number of attention heads
     * @param   head_size         Number of attention heads 
     * 
     * @tparam  T                 Data type of device params
     */

    int h = blockIdx.x;
    T* q = sq + h * head_size;
    T* k = sk + h * head_size;

    int i = threadIdx.x * 2;
    float q0 = q[i];
    float q1 = q[i + 1];
    float k0 = k[i];
    float k1 = k[i + 1];
    float fcr = f_real[i / 2];
    float fci = f_imag[i / 2];
    q[i] = q0 * fcr - q1 * fci;
    q[i + 1] = q0 * fci + q1 * fcr;
    k[i] = k0 * fcr - k1 * fci;
    k[i + 1] = k0 * fci + k1 * fcr;
}

__global__ void softmax_kernel(half* __restrict__ arr, int num_heads, int* pPos) {
    __shared__ float att[MAX_SEQ_LEN_SMEM_KERNEL];
    int h = blockIdx.x;
    int tid = threadIdx.x;
    int step = blockDim.x;
    int size = *pPos + 1;

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
        arr[h * size + t] = (half)(att[t] / sum);
}

__global__ void softmax_kernel_no_smem(half* arr, int num_heads, int* pPos) {
    int h = blockIdx.x;
    int tid = threadIdx.x;
    int step = blockDim.x;
    int size = *pPos + 1;

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
        arr[h * size + i] = (half)val;
        sum += val;
    }

    sum = BlockReduce(temp).Sum(sum);
    if (threadIdx.x == 0)
        shared_val = sum;
    __syncthreads();
    sum = shared_val;

    // normalize and write the result
    for (int t = tid; t < size; t += step)
        arr[h * size + t] = (half)(float(arr[h * size + t]) / sum);
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

// Each block processes a single head
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
__global__ void silu_element_wise_mul_kernel(T* dest, T* src, int size) {
     /**
     * Kernel function to run SiLU element wise multiplication  
     *
     * @param   dest      pointer to src data   
     * @param   src       pointer to dest data
     * @param   size      Number of elements
     * 
     * @tparam  T         Data type of device params
     */

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float val = (float)dest[i];
        val *= 1.0f / (1.0f + expf(-val));
        val *= (float)src[i];
        dest[i] = (T)val;
    }
}

// ----------------------------------------------------------------------------
// Transformer and RunState structs, and related memory management

struct Config{
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length

    friend std::ostream& operator <<(std::ostream& os, Config const& c)
    {
        return os << "dim: (" << c.dim << ")\n"
                  << "hidden_dim: (" << c.hidden_dim << ")\n"
                  << "n_layers: (" << c.n_layers << ")\n"
                  << "n_heads: (" << c.n_heads  << ")\n"
                  << "n_kv_heads: (" <<  c.n_kv_heads << ")\n"
                  << "vocab_size: (" << c.vocab_size << ")\n"
                  << "seq_length: (" << c.seq_len << ")\n";
    }
};

template<typename T>
struct TransformerWeights{
    // token embedding table
    T token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    T rms_att_weight; // (layer, dim) rmsnorm weights
    T rms_ffn_weight; // (layer, dim)
    // weights for matmuls
    T wq; // (layer, dim, dim)
    T wk; // (layer, dim, dim)
    T wv; // (layer, dim, dim)
    T wo; // (layer, dim, dim)
    // weights for ffn
    T w1; // (layer, hidden_dim, dim)
    T w2; // (layer, dim, hidden_dim)
    T w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    T rms_final_weight; // (dim,)
    // freq_cis for RoPE relatively positional embeddings
    T freq_cis_real; // (seq_len, dim/2)
    T freq_cis_imag; // (seq_len, dim/2)
    T wcls;
};


template<typename T1, typename T2>
struct RunState{
    // current wave of activations
    T1 x; // activation at current time stamp (dim,)
    T1 xb; // same, but inside a residual branch (dim,)
    T1 xb2; // an additional buffer just for convenience (dim,)
    T1 hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    T1 hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    T1 q; // query (dim,)
    T1 k; // key (dim,)
    T1 v; // value (dim,)
    T1 att; // buffer for scores/attention values (seq_len,)
    T1 logits_gpu; // output logits gpu
    
    // kv cache
    T1 key_cache;   // (layer, seq_len, dim)
    T1 value_cache; // (layer, seq_len, dim)
    // host side float32 accuracy 
    T2 logits; // output logits cpu
    T2 logits_temp;
};

template<typename T1, typename T2>
void malloc_run_state(RunState<T1, T2>* s, Config* p) {
    /**
     * Function to allocate device memory
     *
     * @param   s          Address of RunState struct 
     * @param   p          Address of Config struct 
     *
     * @tparam  T1         Data type of device params
     * @tparam  T2         Data type of host params
     */

    cudaMalloc((void**)&s->x, p->dim * sizeof(T1));
    cudaMalloc((void**)&s->xb, p->dim * sizeof(T1));
    cudaMalloc((void**)&s->xb2, p->dim * sizeof(T1));
    cudaMalloc((void**)&s->hb, p->hidden_dim * sizeof(T1));
    cudaMalloc((void**)&s->hb2, p->hidden_dim * sizeof(T1));
    cudaMalloc((void**)&s->q, p->dim * sizeof(T1));
    cudaMalloc((void**)&s->k, p->dim * sizeof(T1));
    cudaMalloc((void**)&s->v, p->dim * sizeof(T1));
    cudaMalloc((void**)&s->logits_gpu, p->vocab_size * sizeof(T1));
    cudaMalloc((void**)&s->key_cache, p->n_layers * p->seq_len * p->dim * sizeof(T1));    // potentially huge allocs
    cudaMalloc((void**)&s->value_cache, p->n_layers * p->seq_len * p->dim * sizeof(T1));
    cudaMalloc((void**)&s->logits_temp, p->vocab_size * sizeof(float));
    s->logits = (float*)malloc(p->vocab_size * sizeof(float));

    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
        || !s->k || !s->v || !s->logits || !s->key_cache
        || !s->value_cache || !s->logits_gpu) {
        printf("malloc failed!\n");
        exit(1);
    }
}

template<typename T1, typename T2>
void free_run_state(RunState<T1, T2>* s) {
    /**
     * Function to free device memory for RunState struct 
     *
     * @param   s          Address of RunState struct  
     *
     * @tparam  T1         Data type of device params
     * @tparam  T2         Data type of host params
     */

    cudaFree(s->x);
    cudaFree(s->xb);
    cudaFree(s->xb2);
    cudaFree(s->hb);
    cudaFree(s->hb2);
    cudaFree(s->q);
    cudaFree(s->k);
    cudaFree(s->v);
    cudaFree(s->logits_gpu);
    cudaFree(s->logits_temp);
    free(s->logits);
    cudaFree(s->key_cache);
    cudaFree(s->value_cache);
}

template<typename T>
void malloc_weights(TransformerWeights<T>* w, Config* p, int shared_weights) {
    /**
     * Function to allocate device memory for TransformerWeights struct 
     *
     * @param   w          Address of TransformerWeights struct 
     * @param   p          Address of Config struct 
     * @param   shared_weights    Indicator of whether weights are shared 
     *
     * @tparam  T1         Data type of device params
     * @tparam  T2         Data type of host params
     */
    cudaMalloc((void**)&w->token_embedding_table, p->vocab_size * p->dim * sizeof(T));
    cudaMalloc((void**)&w->rms_att_weight, p->n_layers * p->dim * sizeof(T));
    cudaMalloc((void**)&w->rms_ffn_weight, p->n_layers * p->dim * sizeof(T));
    cudaMalloc((void**)&w->wq, p->n_layers * p->dim * p->dim * sizeof(T));
    cudaMalloc((void**)&w->wk, p->n_layers * p->dim * p->dim * sizeof(T));
    cudaMalloc((void**)&w->wv, p->n_layers * p->dim * p->dim * sizeof(T));
    cudaMalloc((void**)&w->wo, p->n_layers * p->dim * p->dim * sizeof(T));
    cudaMalloc((void**)&w->w1, p->n_layers * p->hidden_dim * p->dim * sizeof(T));
    cudaMalloc((void**)&w->w2, p->n_layers * p->dim * p->hidden_dim * sizeof(T));
    cudaMalloc((void**)&w->w3, p->n_layers * p->hidden_dim * p->dim * sizeof(T));
    cudaMalloc((void**)&w->rms_final_weight, p->dim * sizeof(T));
    int head_size = p->dim / p->n_heads;
    cudaMalloc((void**)&w->freq_cis_real, p->seq_len * head_size / 2 * sizeof(T));
    cudaMalloc((void**)&w->freq_cis_imag, p->seq_len * head_size / 2 * sizeof(T));

    if (shared_weights)
        w->wcls = w->token_embedding_table;
    else
        cudaMalloc((void**)&w->wcls, p->vocab_size * p->dim * sizeof(T));

    // ensure all mallocs went fine
    if (!w->token_embedding_table || !w->rms_att_weight || !w->rms_ffn_weight
        || !w->wq || !w->wk || !w->wv || !w->wo || !w->w1 || !w->w2 || !w->w3 ||
        !w->rms_final_weight || !w->freq_cis_real || !w->freq_cis_imag || !w->wcls) {
        printf("malloc failed!\n");
        exit(1);
    }
}

template<typename T>
void free_weights(TransformerWeights<T>* w, int shared_weights) {
    /**
     * Function to free device memory for TransformerWeights struct 
     *
     * @param   w                 Address of TransformerWeights struct 
     * @param   shared_weights    Indicator of whether weights are shared 
     *
     * @tparam  T                 Data type of device params
     */

    cudaFree(w->token_embedding_table);
    cudaFree(w->rms_att_weight);
    cudaFree(w->rms_ffn_weight);
    cudaFree(w->wq);
    cudaFree(w->wk);
    cudaFree(w->wv);
    cudaFree(w->wo);
    cudaFree(w->w1);
    cudaFree(w->w2);
    cudaFree(w->w3);
    cudaFree(w->rms_final_weight);
    cudaFree(w->freq_cis_real);
    cudaFree(w->freq_cis_imag);
    if (!shared_weights)
        cudaFree(w->wcls);
}

int divUp(int a, int b) {
    /**
     * Function to perform ceil(a,b)
     *
     * @param   a         Numerator
     * @param   b         Denominator 
     */

    return (a - 1) / b + 1;
}

int memcpyToGpu(void *w, int elements, FILE* f, void *scratchCpu, void *scratchGpu, int weight_quant) {
     /**
     * Function to free device memory for TransformerWeights struct 
     *
     * @param   w                 void weight pointer, can hold an address of weight of any data type
     * @param   elements          number of elements 
     * @param   scratchCpu        void pointer for copying data from host to device
     * @param   scratchGpu        void pointer for copying data from host to device
     */

    int count = fread(scratchCpu, sizeof(float), elements, f);
    if (count != elements) return 1;
    // copy and convert fp32->fp16
    cudaMemcpyAsync(scratchGpu, scratchCpu, sizeof(float) * elements, cudaMemcpyHostToDevice);
    if (weight_quant == 1){
        // convert_fp32_to_fp16 
        convert<half> <<<divUp(elements, 256), 256 >>> ((half*)w, (float*)scratchGpu, elements);
    } else{
        convert<float> <<<divUp(elements, 256), 256 >>> ((float*)w, (float*)scratchGpu, elements);
    }
    return 0;
}

// ----------------------------------------------------------------------------
// initialization: read from checkpoint
template <typename T>
int memory_map_weights(TransformerWeights<T>* w, Config* p, FILE* f, int shared_weights, int weight_quant_num) {
    /**
     * Function to free device memory for TransformerWeights struct 
     *
     * @param   w                 Address of TransformerWeights struct 
     * @param   p                 Address of Config struct 
     * @param   f                 Address of char * filename 
     * @param   shared_weight     indicator of whether weights are shared
     * 
     * @tparam  T                 Data type of device params
     */
    
    size_t scratch_size = p->n_layers * std::max(p->dim, p->hidden_dim) * p->dim;
    scratch_size = std::max((size_t)p->vocab_size * p->dim, scratch_size);
    scratch_size *= sizeof(float);
    void* scratchCpu = malloc(scratch_size);
    void* scratchGpu = nullptr;
    cudaMalloc(&scratchGpu, scratch_size);
    if (memcpyToGpu(w->token_embedding_table, p->vocab_size * p->dim, f, scratchCpu, scratchGpu, weight_quant_num)) return 1;
    if (memcpyToGpu(w->rms_att_weight, p->n_layers * p->dim, f, scratchCpu, scratchGpu,  weight_quant_num)) return 1;
    if (memcpyToGpu(w->wq, p->n_layers * p->dim * p->dim, f, scratchCpu, scratchGpu,  weight_quant_num)) return 1;
    if (memcpyToGpu(w->wk, p->n_layers * p->dim * p->dim, f, scratchCpu, scratchGpu,  weight_quant_num)) return 1;
    if (memcpyToGpu(w->wv, p->n_layers * p->dim * p->dim, f, scratchCpu, scratchGpu,  weight_quant_num)) return 1;
    if (memcpyToGpu(w->wo, p->n_layers * p->dim * p->dim, f, scratchCpu, scratchGpu,  weight_quant_num)) return 1;
    if (memcpyToGpu(w->rms_ffn_weight, p->n_layers * p->dim, f, scratchCpu, scratchGpu,  weight_quant_num)) return 1;
    if (memcpyToGpu(w->w1, p->n_layers * p->dim * p->hidden_dim, f, scratchCpu, scratchGpu,  weight_quant_num)) return 1;
    if (memcpyToGpu(w->w2, p->n_layers * p->hidden_dim * p->dim, f, scratchCpu, scratchGpu, weight_quant_num)) return 1;
    if (memcpyToGpu(w->w3, p->n_layers * p->dim * p->hidden_dim, f, scratchCpu, scratchGpu, weight_quant_num)) return 1;
    if (memcpyToGpu(w->rms_final_weight, p->dim, f, scratchCpu, scratchGpu, weight_quant_num)) return 1;

    int head_size = p->dim / p->n_heads;
    if (memcpyToGpu(w->freq_cis_real, p->seq_len * head_size / 2, f, scratchCpu, scratchGpu,  weight_quant_num)) return 1;
    if (memcpyToGpu(w->freq_cis_imag, p->seq_len * head_size / 2, f, scratchCpu, scratchGpu, weight_quant_num)) return 1;

    if (!shared_weights)
        if (memcpyToGpu(w->wcls, p->vocab_size * p->dim, f, scratchCpu, scratchGpu,  weight_quant_num)) return 1;

    cudaFree(scratchGpu);
    free(scratchCpu);
    return 0;
}

// ----------------------------------------------------------------------------
// neural net blocks
template <typename T>
void accum(T* a, T* b, int size) {
    /**
     * Wrapper Function to run atomic add kernel  
     *
     * @param   a                 Address of matrix/vector a
     * @param   b                 Address of matrix/vector b
     * @param   size              Number of elements 
     * 
     * @tparam  T                 Data type of device params
     */
    int blocks = divUp(size, 256);
    element_wise_add_kernel << <blocks, 256 >> > (a, b, size);
}

template <typename T>
void rmsnorm(T* o, T* x, T* weight, int size) {
    /**
     * Wrapper Function to run atomic add kernel  
     *
     * @param   o                 Address of output vector 
     * @param   x                 Address of state vector 
     * @param   weight            Address of weight matrix
     * @param   size              Number of elements 
     * 
     * @tparam  T                 Data type of device params
     */
    int elementsPerThread = divUp(size, 1024);
    rmsnorm_kernel <<<1, 1024 >>> (o, x, weight, size, elementsPerThread);
}

void softmax(float* x, int size) {
    /**
     * Utility Function to convert float into probability
     *
     * @param   x                 Address of state vector 
     * @param   size              Number of elements 
     * 
     */

    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

template <typename T>
void matmul(T* xout, T* x, T* w, int n, int d) {
    /**
     * Wrapper Function to run mat vec multiplication kernel  
     *
     * @param   xout              Address of output vector 
     * @param   x                 Address of state vector 
     * @param   weight            Address of weight matrix
     * @param   size              Number of elements 
     * 
     * @tparam  T                 Data type of device params
     */
    int serialElements = divUp(n, 32);
    dim3 block_dim(32, 4);
    int blocks = divUp(d, 4);
    mat_vec_kernel <<<blocks, block_dim >>> (xout, x, w, n, d, serialElements);
}

template <typename T>
void RoPERotation(T *q, T *k, T *f_real, T *f_imag, int num_heads, int head_size) {
    /**
     * Wrapper Function to run RoPERotation_kernel
     *
     * @param   q                 Address of Q vector 
     * @param   k                 Address of K vector 
     * @param   f_real            Address of f_real
     * @param   f_img             Address of f_img  
     * @param   num_heads         Number of attention heads
     * @param   head_size         Number of attention heads 
     * 
     * @tparam  T                 Data type of device params
     */
    RoPERotation_kernel <<<num_heads, head_size / 2 >>> (q, k, f_real, f_imag, num_heads, head_size);
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

template <typename T>
void siluElementwiseMul(T *hb, T *hb2, int size) {
    /**
     * Wrapper Function to run SiLU kernel
     *
     * @param   hb                Address of buffer for hidden dimension in the ffn (hidden_dim,)
     * @param   hb2               Address of buffer for hidden dimension in the ffn (hidden_dim,)
     * @param   size              Number of elements
     * 
     * @tparam  T                 Data type of device params
     */
 
   silu_element_wise_mul_kernel <<<divUp(size, 256), 256 >>> (hb, hb2, size);
}

template <typename T1, typename T2>
void transformer(int token, int pos, Config* p, RunState<T1*, T2*>* s, TransformerWeights<T1*>* w) {
    /**
     * Wrapper Function to run atomic add kernel  
     *
     * @param   token            Index of token in Llama-2 sentencepiece, 1 = BOS token 
     * @param   pos              position in the sequence 
     * @param   p                Address of buffer for struct Config 
     * @param   s                Address of buffer for struct Runstate 
     * @param   w                Address of buffer for struct TransformerWeights
     * 
     * 
     * @tparam  T1               Data type of device params
     * @tparam  T2               Data type of host params
     */

    T1* x = s->x;
    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x
    T1* content_row = &(w->token_embedding_table[token * dim]);
    cudaMemcpyAsync(x, content_row, dim * sizeof(half), cudaMemcpyDeviceToDevice);

    // pluck out the "pos" row of freq_cis_real and freq_cis_imag
    T1* freq_cis_real_row = w->freq_cis_real + pos * head_size / 2;
    T1* freq_cis_imag_row = w->freq_cis_imag + pos * head_size / 2;

    // forward all the layers
    for (int l = 0; l < p->n_layers; l++) {

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);

        // qkv matmuls for this position
        matmul(s->q, s->xb, w->wq + l * dim * dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l * dim * dim, dim, dim);
        matmul(s->v, s->xb, w->wv + l * dim * dim, dim, dim);

        // apply RoPE rotation to the q and k vectors for each head
        RoPERotation(s->q, s->k, freq_cis_real_row, freq_cis_imag_row, p->n_heads, head_size);

        // save key,value at this time step (pos) to our kv cache
        int loff = l * p->seq_len * dim; // kv cache layer offset for convenience
        T1* key_cache_row = s->key_cache + loff + pos * dim;
        T1* value_cache_row = s->value_cache + loff + pos * dim;
        cudaMemcpyAsync(key_cache_row, s->k, dim * sizeof(T1), cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(value_cache_row, s->v, dim * sizeof(T1), cudaMemcpyDeviceToDevice);

        MultiHeadAttention(s->xb, s->q, s->key_cache, s->value_cache, p->n_heads, head_size, loff, pos+1);

        // final matmul to get the output of the attention
        matmul(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);

        // residual connection back into x
        accum(x, s->xb2, dim);

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(s->hb, s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim);

        // apply F.silu activation on hb and multiply it with hb2
        siluElementwiseMul(s->hb, s->hb2, hidden_dim);

        // final matmul to get the output of the ffn
        matmul(s->xb, s->hb, w->w2 + l * dim * hidden_dim, hidden_dim, dim);

        // residual connection
        accum(x, s->xb, dim);
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    matmul(s->logits_gpu, x, w->wcls, p->dim, p->vocab_size);

    // copy logits from GPU->CPU and convert_fp16_to_fp32
    convert<float> <<<divUp(p->vocab_size, 256), 256 >>> (s->logits_temp, s->logits_gpu, p->vocab_size);
    cudaMemcpy(s->logits, s->logits_temp, p->vocab_size * sizeof(float), cudaMemcpyDeviceToHost);
}

// ----------------------------------------------------------------------------
// byte pair encoding (BPE) tokenizer, encodes strings into tokens so we can prompt

int str_lookup(char *str, char **vocab, int vocab_size) {
     /**
     * Utility Function to find the first perfect match for str in vocab, return its index or -1 if not found add kernel  
     *
     * @param   token            Index of token in Llama-2 sentencepiece, 1 = BOS token 
     * @param   pos              Index of the time steps 
     * @param   p                Address of buffer for struct Config 
     * @param   s                Address of buffer for struct Runstate 
     * @param   w                Address of buffer for struct TransformerWeights
     * 
     * @tparam  T1               Data type of device params
     * @tparam  T2               Data type of host params
     */
    for (int i = 0; i < vocab_size; i++) {
        if (strcmp(str, vocab[i]) == 0) {
            return i;
        }
    }
    return -1;
}

void bpe_encode(char *text, char **vocab, float *vocab_scores, int vocab_size, unsigned int max_token_length, int *tokens, int *n_tokens) {
    /**
     * Utility Function to merge the best consecutive pair each iteration, according the scores in vocab_scores
     *
     * @param   text                    Prompt string 
     * @param   vocab                   Vocabulary 
     * @param   vocab_scores            Score for its corresponding vocabulary, the higher the better
     * @param   max_token_length        Maximum token length
     * @param   n_tokens                Address of buffer for struct TransformerWeights
     */

    // a temporary buffer to merge two consecutive tokens
    char* str_buffer = (char*) malloc((max_token_length*2+1) * sizeof(char)); // *2 for concat, +1 for null terminator

    // first encode every individual byte in the input string
    *n_tokens = 0; // the number of tokens
    for (char *c = text; *c != '\0'; c++) {
        sprintf(str_buffer, "%c", *c);
        int id = str_lookup(str_buffer, vocab, vocab_size);
        if (id == -1) { printf("not good\n"); exit(1);}
        tokens[*n_tokens] = id;
        (*n_tokens)++;
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", vocab[tokens[i]], vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, vocab, vocab_size);
            if (id != -1 && vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }

    free(str_buffer);
}

// ----------------------------------------------------------------------------
// utilities

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    timespec_get(&time, TIME_UTC);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

unsigned int random_u32(unsigned long rng_seed) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    rng_seed ^= rng_seed >> 12;
    rng_seed ^= rng_seed << 25;
    rng_seed ^= rng_seed >> 27;
    return (rng_seed * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32(unsigned long rng_seed) { // random float32 in [0,1)
    return (random_u32(rng_seed) >> 8) / 16777216.0f;
}

int sample(float* probabilities, int n, unsigned long rng_seed) {
    /**
     * Utility Function to sample index from probabilities, they must sum to 1
     *
     * @param   probabilities           pointer to the probabilities 
     * @param   n                       Range of numbers to sample from  
     * @param   rng_seed                Random seed for random number generation 
     */

    
    float r = random_f32(rng_seed);
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (r < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int argmax(float* v, int n) {
     /**
     * Utility Function that returns argmax of v in elements 0..n

     *
     * @param   v                       pointer to the float array
     * @param   n                       Number of elements in the array
     */

    int max_i = 0;
    float max_p = v[0];
    for (int i = 1; i < n; i++) {
        if (v[i] > max_p) {
            max_i = i;
            max_p = v[i];
        }
    }
    return max_i;
}

FILE* load_checkpoint(char* checkpoint){
    /**
     * Function that open and returns the file 'checkpoint'

     *
     * @param   checkpoint     pointer to a sequence of characters that is the model checkpoint filename
     */

    FILE *file = fopen(checkpoint, "rb");
    if (!file) { printf("Couldn't open file %s\n", checkpoint);}
    return file;
}

int load_config(FILE* file, Config* config){
    /**
     * Function that reads and loads config struct  

     *
     * @param   file      pointer to FILE
     * @param   config    pointer to Config struct
     */

    if (fread(config, sizeof(Config), 1, file) != 1) { return 1; }
    // Dump model config
    std::cout << *config << std::endl;
    return 0;
}

template<typename T>
int load_weights(FILE* file, Config* config, TransformerWeights<T>* weights, int shared_weights, int wq_num){
    /**
     * Function that allocates device memory for weights and reads & loads weights from file   

     *
     * @param   file            pointer to FILE
     * @param   weights         pointer to TransformerWeights struct 
     * @param   config          pointer to Config struct
     * @param   shared_weights  indicator of shared weights 
     * 
     * @tparam  T               data type of device params 
     */

    malloc_weights(weights, config, shared_weights);
    if (memory_map_weights(weights, config, file, shared_weights, wq_num)) { return 1; }
    return 0;
}

int load_tokenizer(Config config, FILE* tokenizer_file, char** vocab, float* vocab_scores, unsigned int* max_token_length){
    /**
     * Function that reads & loads tokenizer  
     * 
     * @param   config              pointer to Config struct
     * @param   tokenizer_file      pointer to tokenizer FILE
     * @param   vocab               double pointer to vocabulary
     * @param   vocab_scores        pointer to the score of the corresponding vocabulary given by the tokenizer 
     * @param   max_token_length    maximum length of the sequence of tokens
     */

    // read in the tokenizer.bin file
    if (!tokenizer_file) { printf("couldn't load tokenizer.bin\n"); return 1; }
    if (fread(&max_token_length, sizeof(int), 1, tokenizer_file) != 1) { printf("failed read\n"); Sleep(1000); return 1; }

    int len;
    for (int i = 0; i < config.vocab_size; i++) {
        if (fread(vocab_scores + i, sizeof(float), 1, tokenizer_file) != 1) { printf("failed read\n"); Sleep(1000); return 1;}
        if (fread(&len, sizeof(int), 1, tokenizer_file) != 1) { printf("failed read\n"); Sleep(1000); return 1; }
        vocab[i] = (char *)malloc(len + 1);
        if (fread(vocab[i], len, 1, tokenizer_file) != 1) { printf("failed read\n"); Sleep(1000); return 1; }
        vocab[i][len] = '\0'; // add the string terminating token
    }
    return 0;
}

template <typename T1, typename T2>
void generate_tokens(char *prompt, char* checkpoint, int steps, char** vocab, float* vocab_scores, Config config, 
TransformerWeights<T1> weights, RunState<T1, T2> state, unsigned int max_token_length, float temperature, int shared_weights, int weight_quant_num, unsigned long rng_seed){
    /**
     * Function to load config, model and tokenizer, and then generate tokens with different param precision on deivce 

     * @param   prompt                  pointer to the prompt chars
     * @param   checkpoint              pointer to the char seq (filename of the checkpoint e.g. Model.bin)
     * @param   steps                   the number of steps
     * @param   config                  config Struct 
     * @param   vocab                   double pointer to vocab char seq
     * @param   vocab_scores            pointer to vocab score
     * @param   weights                 templated TransformerWeights struct for different precision 
     * @param   state                   templated RunState struct for different precision
     * @param   max_token_length        maximum length of the sequence of tokens
     * @param   temperature             softmax temperature parameter
     * @param   shared_weights          Indicator of whether weights are shared
     * @param   weight_quant_num        Indicator of param precision on device, currently supporting 0 for full precision, 1 for half precision (default)
     * @param   rng_seed                Used to generate random number 
     * 
     * @tparam  T1                      Data type for params on device  
     * @tparam  T2                      Data type for params on host 
     */
    
    
    // process the prompt, if any
    int *prompt_tokens = NULL;
    int num_prompt_tokens = 0;
    if (prompt != NULL) {
        prompt_tokens = (int*)malloc(config.seq_len * sizeof(int));
        bpe_encode(prompt, vocab, vocab_scores, config.vocab_size, max_token_length, prompt_tokens, &num_prompt_tokens);
    }

    // start the main loop
    long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = 1;   // init with token 1 (=BOS), as done in Llama-2 sentencepiece tokenizer
    int pos = 0;     // position in the sequence
    printf("<s>\n"); // explicit print the initial BOS token for stylistic symmetry reasons

    FILE * File;
    std::ostringstream stringStream;    
    stringStream << "benchmark_results/" << "gpu_" << checkpoint << "_output.txt";
    std::string filename = stringStream.str();

    File = fopen(filename.c_str(), "w+");
    while (pos < steps) {

        // forward the transformer to get logits for the next token
        transformer(token, pos, &config, &state, &weights);

        if(pos < num_prompt_tokens) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos];
        } else {
            // sample the next token
            if (temperature == 0.0f) {
                // greedy argmax sampling: take the token with the highest probability
                next = argmax(state.logits, config.vocab_size);
            } else {
                // apply the temperature to the logits
                for (int q=0; q<config.vocab_size; q++) { state.logits[q] /= temperature; }
                // apply softmax to the logits to get the probabilities for next token
                softmax(state.logits, config.vocab_size);
                // we sample from this distribution to get the next token
                next = sample(state.logits, config.vocab_size, rng_seed);
            }
        }

        // following BOS token (1), sentencepiece decoder strips any leading whitespace 
        char *token_str = (token == 1 && vocab[next][0] == ' ') ? vocab[next]+1 : vocab[next];
        printf("%s", token_str);
        fflush(stdout);
        writeToFile(token_str);
        writeToFile("\n");
        if (next == 2) break; // break if EOS token is reached

        // advance forward
        token = next;
        pos++;
        // init our timer here because the first iteration could be slow
        if (start == 0) { start = time_in_ms(); }
    }

    // report achieved tok/s
    long end = time_in_ms();
    double time = (end - start) / 1000.0;
    int timed_tokens = pos - 1;
    printf("\nachieved tok/s: %f. Tokens: %d, seconds: %g\n", timed_tokens / time, timed_tokens, time);
    
    writeToFile("precision: %d\n", weight_quant_num);
    writeToFile("achieved tok/s: (%f). Tokens: (%d), seconds: (%g)\n", timed_tokens / time, timed_tokens, time);
    std::ostringstream fs;
    fs << config << std::endl;
    writeToFile(fs.str().c_str());
    fclose(File);
    
    // memory cleanup
    free_run_state(&state);
    free_weights(&weights, shared_weights);
    for (int i = 0; i < config.vocab_size; i++) { free(vocab[i]); }
    free(vocab);
    free(vocab_scores);
    if (prompt_tokens != NULL) free(prompt_tokens);
    Sleep(1000);
}

int run_inference_fp16(char* checkpoint, char* prompt, float temperature, int steps, unsigned long rng_seed, int weight_quant_num){
    /**
     * Function to load config, model and tokenizer, and then generate tokens with param precision 'half (fp16)' on deivce 

     * @param   checkpoint              pointer to the filename of the checkpoint e.g. Model.bin
     * @param   prompt                  pointer to prompt chars
     * @param   temperature             Softmax temperature parameter
     * @param   state                   templated RunState struct for different precision
     * @param   weights                 templated TransformerWeights struct for different precision 
     * @param   config                  config Struct 
     * @param   shared_weights          Indicator of whether weights are shared
     * @param   rng_seed                Used to generate random number 
     * @param   weight_quant_num        Indicator of param precision on device, currently supporting 0 for full precision, 1 for half precision (default)
     * 
     * @tparam  T1                      Data type for params on device  
     * @tparam  T2                      Data type for params on host 
     */
    Config config;
    TransformerWeights<half*> weights;
    RunState<half*, float*> state;
    FILE* file = load_checkpoint(checkpoint);
    // read in the config header
    if (load_config(file, &config)) {printf("load config failed\n"); Sleep(1000); return 1;}
    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    int shared_weights = config.vocab_size > 0 ? 1 : 0;
    config.vocab_size = abs(config.vocab_size);
    // read in the Transformer weights
    if (load_weights(file, &config, &weights, shared_weights, weight_quant_num)) {printf("load weights failed \n"); Sleep(1000); return 1;}
    fclose(file);
 
    // right now we cannot run for more than config.seq_len steps
    if (steps <= 0 || steps > config.seq_len) { steps = config.seq_len; }
    FILE *tokenizer_file = fopen("tokenizer.bin", "rb");
    unsigned int max_token_length=0; 
    char** vocab = (char**)malloc(config.vocab_size * sizeof(char*));
    float* vocab_scores = (float*)malloc(config.vocab_size * sizeof(float));
    // (Config config, FILE* tokenizer_file, char** vocab, float* vocab_scores, unsigned int* max_token_length)
    if (load_tokenizer(config, tokenizer_file, vocab, vocab_scores, &max_token_length)) {printf("load tokenizer failed\n"); Sleep(1000); return 1;}
    fclose(tokenizer_file);
    // create and init the application RunState
    malloc_run_state(&state, &config);
    generate_tokens(prompt, checkpoint, steps, vocab, vocab_scores, config, weights, state, max_token_length, temperature, shared_weights, weight_quant_num, rng_seed);
   
    return 0;
}

int run_inference_fp32(char* checkpoint, char* prompt, float temperature, int steps, unsigned long rng_seed, int weight_quant_num){
    /**
     * Function to load config, model and tokenizer, and then generate tokens with param precision 'full (fp32)' on deivce 

     * @param   checkpoint              pointer to filename of the checkpoint e.g. Model.bin
     * @param   prompt                  pointer to prompt chars
     * @param   temperature             Softmax temperature parameter
     * @param   state                   templated RunState struct for different precision
     * @param   weights                 templated TransformerWeights struct for different precision 
     * @param   config                  config Struct 
     * @param   shared_weights          Indicator of whether weights are shared
     * @param   rng_seed                Used to generate random number 
     * @param   weight_quant_num        Indicator of param precision on device, currently supporting 0 for full precision, 1 for half precision (default)
     * 
     * @tparam  T1                      Data type for params on device  
     * @tparam  T2                      Data type for params on host 
     */
    Config config;
    TransformerWeights<float*> weights;
    RunState<float*, float*> state;
    FILE* file = load_checkpoint(checkpoint);
    // read in the config header
    if (load_config(file, &config)) {printf("load config failed\n"); Sleep(1000); return 1;}
    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    int shared_weights = config.vocab_size > 0 ? 1 : 0;
    config.vocab_size = abs(config.vocab_size);
    // read in the Transformer weights
    if (load_weights(file, &config, &weights, shared_weights, weight_quant_num)) {printf("load weights failed \n"); Sleep(1000); return 1;}
    fclose(file);
 
    // right now we cannot run for more than config.seq_len steps
    if (steps <= 0 || steps > config.seq_len) { steps = config.seq_len; }
    FILE *tokenizer_file = fopen("tokenizer.bin", "rb");
    unsigned int max_token_length=0; char** vocab=nullptr; float* vocab_scores=nullptr;
    // (Config config, FILE* tokenizer_file, char** vocab, float* vocab_scores, unsigned int* max_token_length)
    if (load_tokenizer(config, tokenizer_file, vocab, vocab_scores, &max_token_length)) {printf("load tokenizer failed\n"); Sleep(1000); return 1;}
    fclose(tokenizer_file);

    // create and init the application RunState
    malloc_run_state(&state, &config);
    generate_tokens(prompt, checkpoint, steps, vocab, vocab_scores, config, weights, state, max_token_length, temperature, shared_weights, weight_quant_num, rng_seed);
   
    return 0;
}
// ----------------------------------------------------------------------------

int main(int argc, char *argv[]) {

    char *checkpoint = NULL;  // e.g. out/model.bin
    float temperature = 0.9f; // e.g. 1.0, or 0.0
    int steps = 256;          // max number of steps to run for, 0: use seq_len
    char *prompt = NULL;      // prompt string
    int weight_quant = 1;     // default using half data type for device params 
    // 'checkpoint' is necessary arg
    if (argc < 2) {
        printf("Usage: %s <checkpoint_file> [weight_quant] [steps] [temperature] [prompt]\n", argv[0]);
        return 1;
    }
    if (argc >= 2) {
        checkpoint = argv[1];
    }
    if (argc >= 3) {
        weight_quant = atoi(argv[2]);
        
    }
    if (argc >= 4) {
        steps = atoi(argv[3]);
    }

    if (argc >= 5){
        // optional temperature. 0.0 = (deterministic) argmax sampling. 1.0 = baseline
        temperature = atof(argv[4]);
    }

    if (argc >= 6) {
        prompt = argv[5];
    }

    // seed rng with time. if you want deterministic behavior use temperature 0.0
    unsigned long rng_seed;
    rng_seed = (unsigned int)time(NULL);
    
    // read in the model.bin file
    if (weight_quant == 1){
        printf("using half precision on device\n");
       
        if (run_inference_fp16(checkpoint, prompt, temperature, steps, rng_seed, weight_quant)) return 1;
    } else{
        printf("using full precision on device\n");
       
        if (run_inference_fp32(checkpoint, prompt, temperature, steps, rng_seed, weight_quant)) return 1;
    }
    return 0;

}