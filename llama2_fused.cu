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
    float rope_theta; // theta for the rope rotational embedding

    friend std::ostream& operator <<(std::ostream& os, Config const& c)
    {
        return os << "dim: (" << c.dim << ")\n"
                  << "hidden_dim: (" << c.hidden_dim << ")\n"
                  << "n_layers: (" << c.n_layers << ")\n"
                  << "n_heads: (" << c.n_heads  << ")\n"
                  << "n_kv_heads: (" <<  c.n_kv_heads << ")\n"
                  << "vocab_size: (" << c.vocab_size << ")\n"
                  << "seq_length: (" << c.seq_len << ")\n"
                  << "rope_theta: (" << c.rope_theta << ")\n";
    }
};

template <typename T>
struct PerLayerWeight {
    T* rms_att_weight; // (layer, dim) rmsnorm weights
    T* rms_ffn_weight; // (layer, dim)
    T* wq;
    T* wk;
    T* wv;
    T* wo;
    T* wgate;
    T* wup;
    T* wdown;
};

template <typename T>
struct TransformerWeights {
    // token embedding table
    T* token_embedding_table;    // (vocab_size, dim)
    // classifier weights for the logits, on the last layer
    T* wcls;
    // final rmsnorm
    T* rms_final_weight; // (dim,)
    // Per layer weights
    PerLayerWeight<T>* layers;
    int num_layers;
};

// data shared between CPU and GPU (allocated in host memory)
struct SharedData {
    volatile int pos;         // current token index
    int tokens[MAX_SEQ_LEN];  // seq_len (tokens processed/generated so far) allocated in host memory so that CPU can read this
};

struct Sampler{
    int vocab_size = 0;
    int* indices = nullptr;
    void* tempStorage_scan = nullptr;
    void* tempStorage_sort = nullptr;
    size_t temp_storage_bytes_scan = 0;
    size_t temp_storage_bytes_sort = 0;
    float temperature = 0;
    float topp = 0;
    unsigned long rng_state = 0;
};

template<typename T1, typename T2>
struct RunState{
    // current wave of activations
    T1* x; // activation at current time stamp (dim,)
    T1* xb; // same, but inside a residual branch (dim,)
    T1* xb2; // an additional buffer just for convenience (dim,)
    T1* hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    T1* hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    T1* q; // query (dim,)
    T1* att; // buffer for scores/attention values (n_heads, seq_len)
    T1* logits; // output logits gpu
    
    // kv cache
    T1* key_cache;   // (layer, seq_len, dim)
    T1* value_cache; // (layer, seq_len, dim)

    int* pos;  // GPU copy of the current position (just 1 element)
    SharedData* shared_data;

    T2* logits_array;  // array of output logits used to compute perplexity (seq_len, vocab_size)
};

// hardcoded for llama models
constexpr int bos_token = 1;
constexpr int eos_token = 2;


// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens
// ----------------------------------------------------------------------------

typedef struct {
    char* str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex* sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

// ----------------------------------------------------------------------------
int compare_tokens(const void* a, const void* b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
    // i should have written the vocab_size into the tokenizer file... sigh
    t->vocab_size = vocab_size;
    // malloc space to hold the scores and the strings
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL; // initialized lazily
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    // read in the file
    FILE* file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i] = (char*)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i][len] = '\0'; // add the string terminating token
    }
    fclose(file);
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

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



// Each block processes a single head
template <typename T>
__global__ void RoPERotation_kernel(T* sq, T* sk_base, int num_kv_heads, int head_size, int* pPos, int loff, float rope_theta) {
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

    int pos = *pPos;
    int h = blockIdx.x;
    T* q = sq + h * head_size;
    int i = threadIdx.x;
    int head_dim = (i * 2) % head_size;
    float freq = 1.0f / powf(rope_theta, head_dim / (float)head_size);
    float val = pos * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    float q0 = q[i];
    float q1 = q[i + head_size / 2];
    q[i] = q0 * fcr - q1 * fci;
    q[i + head_size / 2] = q0 * fci + q1 * fcr;
    if (h < num_kv_heads) {
        T* sk = sk_base + loff + pos * num_kv_heads * head_size;
        T* k = sk + h * head_size;
        float k0 = k[i];
        float k1 = k[i + head_size / 2];
        k[i] = k0 * fcr - k1 * fci;
        k[i + head_size / 2] = k0 * fci + k1 * fcr;
    }
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

template <typename T>
__global__ void softmax_logits_kernel(T* __restrict__ logits, int size, float temperature, int *indices) {
    int tid = threadIdx.x;
    int step = blockDim.x;

    
    for (int t = tid; t < size; t += step)
    {
        // first just write the indices array
        indices[t] = t;

        // divide by temperature
        float val = (float)logits[t];
        val /= temperature;
        logits[t] = (T)val;
    }
    __syncthreads();

    // Compute the softmax
    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_val;

    // find max value (for numerical stability)
    float max_val = tid < size ? ((float)logits[tid]) : -FLT_MAX;
    for (int i = tid + step; i < size; i += step)
        if ((float)logits[i] > max_val)
            max_val = logits[i];

    max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
    if (threadIdx.x == 0)
        shared_val = max_val;
    __syncthreads();
    max_val = shared_val;

    // exp and sum
    float sum = 0.0f;
    for (int i = tid; i < size; i += step) {
        float v = expf(float(logits[i]) - max_val);
        logits[i] = (T)v;
        sum += v;
    }

    sum = BlockReduce(temp).Sum(sum);
    if (threadIdx.x == 0)
        shared_val = sum;
    __syncthreads();
    sum = shared_val;

    // normalize and write the result
    for (int t = tid; t < size; t += step)
        logits[t] = (T)(float(logits[t]) / sum);
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



// Here we make use of shared memory to achieve better memory access pattern, and transpose a 32x32 chunk of the matrix on the fly
// Again used only by the MHA block
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

// find the index in the array that crosses top-p threshold
// __global__ void sample_top_p_kernel(half* sorted_logits_prefix_sum, int* indices, int n, float top_p_threshold, int* result, volatile int* pPos, int* pPosGpu)
template <typename T>
__global__ void sample_top_p_kernel(T* sorted_logits_prefix_sum, int* indices, int n, float top_p_threshold, int* result, volatile int* pPos, int* pPosGpu)
{
    int tid = threadIdx.x;
    int step = blockDim.x;

    int min_index = n - 1;

    for (int t = tid; t < n; t += step) {
        if ((float)(sorted_logits_prefix_sum[t]) >= top_p_threshold) {
            if (t < min_index) {
                min_index = t;
            }
        }
    }

    // find the min across the block
    using BlockReduce = cub::BlockReduce<int, 1024>;
    __shared__ typename BlockReduce::TempStorage temp;
    int min_index_global = BlockReduce(temp).Reduce(min_index, cub::Min());
    if (threadIdx.x == 0)
    {
        int token_pos = *pPos;
        token_pos++;
        result[token_pos] = indices[min_index_global];

        // update the token indices
        *pPos = token_pos;
        // *pPosGpu = token_pos;
    }
}

template <typename T>
__global__ void argmax_kernel(T* __restrict__ x, int size, int* result, volatile int* pPos, int* pPosGpu, bool write_token) {
    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_val;

    int tid = threadIdx.x;
    int step = blockDim.x;

    // find local max value and its position
    float max_val = tid < size ? (float)x[tid] : -INFINITY;
    int   max_pos = tid < size ? tid : 0;
    for (int i = tid + step; i < size; i += step) {
        if ((float)x[i] > max_val) {
            max_val = x[i];
            max_pos = i;
        }
    }

    // find the global max value
    float global_max_val;
    global_max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
    if (threadIdx.x == 0)
        shared_val = global_max_val;
    __syncthreads();
    global_max_val = shared_val;

    // possibility of race condition here, so we first write it to shared memory variable and then have just one thread to update the pointers.
    __shared__ int global_max_pos;
    if (max_val == global_max_val) {
        global_max_pos = max_pos;
    }
    __syncthreads();

    // write next token to the current token location
    if (threadIdx.x == 0) {
        int token_pos = *pPos;
        token_pos++;

        if (write_token)
            result[token_pos] = global_max_pos;

        // update the token indices (unblocks the CPU)
        *pPos = token_pos;
        *pPosGpu = token_pos;
    }
}


// sample the token given the logits and some hyperparameters
template <typename T1, typename T2>
// sample the token given the logits and some hyperparameters
void sample(Sampler* sampler, RunState<T1, T2>* s, bool gen_token, cudaStream_t stream) {
    // flip a (float) coin (this is our source of entropy for sampling)
    float coin = random_f32(sampler->rng_state);

    if (sampler->temperature == 0.0f || !gen_token) {
        // greedy argmax sampling: take the token with the highest probability
        argmax_kernel <<<1, 1024, 0, stream >>> (s->logits, sampler->vocab_size, &(s->shared_data->tokens[0]), &(s->shared_data->pos), s->pos, gen_token);
    }
    else {
        // apply the temperature to the logits, and then perform softmax
        softmax_logits_kernel <<<1, 1024, 0, stream >>> (s->logits, sampler->vocab_size, sampler->temperature, sampler->indices);

        float threshold = 0.0f;
        // we sample from this distribution to get the next token
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            threshold = coin;
        }
        else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            if (sampler->temp_storage_bytes_sort == 0) {
                cub::DeviceRadixSort::SortPairsDescending(sampler->tempStorage_sort, sampler->temp_storage_bytes_sort, s->logits, s->logits, sampler->indices, sampler->indices,
                    sampler->vocab_size, 0, sizeof(T1) * 8, stream);
                cudaMalloc(&sampler->tempStorage_sort, sampler->temp_storage_bytes_sort);
            }

            cub::DeviceRadixSort::SortPairsDescending(sampler->tempStorage_sort, sampler->temp_storage_bytes_sort, s->logits, s->logits, sampler->indices, sampler->indices, 
                sampler->vocab_size, 0, sizeof(T1) * 8, stream);
            threshold = coin * sampler->topp;
        }

        // Sample from the predicted probability distribution
        if (sampler->temp_storage_bytes_scan == 0) {
            cub::DeviceScan::InclusiveSum(sampler->tempStorage_scan, sampler->temp_storage_bytes_scan, s->logits, s->logits, sampler->vocab_size, stream);
            cudaMalloc(&sampler->tempStorage_scan, sampler->temp_storage_bytes_scan);
        }
        cub::DeviceScan::InclusiveSum(sampler->tempStorage_scan, sampler->temp_storage_bytes_scan, s->logits, s->logits, sampler->vocab_size, stream);

        sample_top_p_kernel <<<1, 1024, 0, stream >>> (s->logits, sampler->indices, sampler->vocab_size, threshold, &(s->shared_data->tokens[0]), &(s->shared_data->pos), s->pos);
    }
}

template <typename T>
void MultiHeadAttentionFused(T *output, T *q, T *key_cache, T* value_cache, T *att, int num_heads, int head_size, int kv_mul, int max_seq_len, int* ppos) {
    int pos = *ppos;
    int dim = head_size * num_heads;
    // 1. Get attention scores
    int serialElements = divUp(head_size, 32);
    dim3 block_dim(32, 32);
    dim3 grid_dim1(divUp(max_seq_len, 32), num_heads);      // using max_seq_len instead of real seq_len here has measurable impact on perf (2%) :-/
    mat_vec_kernel_MHA <<< grid_dim1, block_dim, 0, stream >>> (att, q, key_cache, head_size, serialElements, head_size, head_size, dim / kv_mul, 1.0 / sqrt(head_size), pos, kv_mul);

    // 2. Run softmax kernel
    if (max_seq_len <= MAX_SEQ_LEN_SMEM_KERNEL)
        softmax_kernel <<< num_heads, 1024, 0, stream >>> (att, num_heads, pos);
    else
        softmax_kernel_no_smem <<< num_heads, 1024, 0, stream >>> (att, num_heads, pos);

    // 3. weighted sum of the values to get the final result
    dim3 grid_dim2(divUp(head_size, 32), num_heads);
    vec_mat_kernel_MHA <<< grid_dim2, block_dim, 0, stream >>> (output, att, value_cache, head_size, ppos, head_size, head_size, dim / kv_mul, kv_mul);
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
     * @param   seq_len           current position in the sequence  
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



template<typename T>
void alloc_gpu_space(T* w, int n, size_t size){
    cudaMalloc((void**)&w,  n * size);
}

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
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    alloc_gpu_space(s->x, p->dim, sizeof(T1));
    alloc_gpu_space(s->xb, p->dim, sizeof(T1));
    alloc_gpu_space(s->xb2, p->dim, sizeof(T1));
    alloc_gpu_space(s->hb, p->hidden_dim, sizeof(T1));
    alloc_gpu_space(s->hb2, p->hidden_dim, sizeof(T1));
    alloc_gpu_space(s->q, p->dim, sizeof(T1));
    alloc_gpu_space(s->att, p->n_heads * p->dim, sizeof(T1));
    alloc_gpu_space(s->logits, p->vocab_size, sizeof(T1));
    alloc_gpu_space(s->key_cache, p->n_layers * p->seq_len * kv_dim, sizeof(T1));    // potentially huge allocs
    alloc_gpu_space(s->value_cache, p->n_layers * p->seq_len * kv_dim, sizeof(T1));
   
    cudaMalloc((void**)&s->pos, sizeof(int));
    cudaMallocHost((void**)&s->shared_data, sizeof(SharedData));

    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q || !s-> att || !s->key_cache
        || !s->value_cache || !s->logits) {
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
    cudaFree(s->att);
    cudaFree(s->logits);
    cudaFree(s->key_cache);
    cudaFree(s->value_cache);
    cudaFreeHost(s->shared_data);
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

    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    alloc_gpu_space(w->token_embedding_table, p->vocab_size*p->dim, sizeof(T));
    w->layers = (PerLayerWeight<T>*)malloc(p->n_layers * sizeof(PerLayerWeight<T>));
    w->num_layers = p->n_layers;
    for (int l = 0; l < p->n_layers; l++)
    {
        PerLayerWeight<T>* layer = &(w->layers[l]);
        alloc_gpu_space(layer->rms_att_weight, p->dim, sizeof(T));
        alloc_gpu_space(layer->rms_ffn_weight, p->dim, sizeof(T));
        alloc_gpu_space(layer->wq, p->dim * p->dim, sizeof(T));
        alloc_gpu_space(layer->wk, p->n_layers * p->dim * kv_dim, sizeof(T));
        alloc_gpu_space(layer->wv, p->n_layers * p->dim * kv_dim, sizeof(T));
        alloc_gpu_space(layer->wo, p->n_layers * p->dim * p->dim, sizeof(T));
        alloc_gpu_space(layer->wgate, p->n_layers * p->hidden_dim * p->dim, sizeof(T));
        alloc_gpu_space(layer->wup, p->n_layers * p->dim * p->hidden_dim, sizeof(T));
        alloc_gpu_space(layer->wdown, p->n_layers * p->hidden_dim * p->dim, sizeof(T));
    }
        
    alloc_gpu_space(w->rms_final_weight, p->dim, sizeof(T));
    alloc_gpu_space(w->wcls, p->vocab_size * p->dim, sizeof(T));

   
    // ensure all mallocs went fine
    if (!w->token_embedding_table || !w->layers ||
        !w->rms_final_weight || !w->wcls) {
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
    cudaFree(w->rms_final_weight);
    cudaFree(w->wcls);
    for (int l = 0; l < w->num_layers; l++) {
        PerLayerWeight<T>* layer = &(w->layers[l]);
        cudaFree(&layer->rms_att_weight);
        cudaFree(&layer->rms_ffn_weight);
        cudaFree(&layer->wq);
        cudaFree(&layer->wk);
        cudaFree(&layer->wv);
        cudaFree(&layer->wo);
        cudaFree(&layer->wgate);
        cudaFree(&layer->wup);
        cudaFree(&layer->wdown);
    }
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
    
    size_t scratch_size = std::max(p->vocab_size, p->hidden_dim) * p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    scratch_size *= sizeof(half);
    void* scratchCpu = malloc(scratch_size);

    printf("\nLoading Weights... ");

    memoryMapWeight(w->token_embedding_table, f, p->vocab_size * p->dim * sizeof(half), scratchCpu);
    memoryMapWeight(w->wcls, f, p->vocab_size * p->dim * sizeof(half), scratchCpu);
    memoryMapWeight(w->rms_final_weight, f, p->dim * sizeof(half), scratchCpu);

    // upload decoder block weight for each layer
    for (int i = 0; i < p->n_layers; i++) {
        memoryMapWeight(w->layers[i].wq, f, p->dim*p->dim, scratchCpu);
        memoryMapWeight(w->layers[i].wk, f, p->dim* kv_dim, scratchCpu);
        memoryMapWeight(w->layers[i].wv, f, p->dim* kv_dim, scratchCpu);
        memoryMapWeight(w->layers[i].wo, f, p->dim* p->dim, scratchCpu);

        memoryMapWeight(w->layers[i].wup, f, p->dim* p->hidden_dim, scratchCpu);
        memoryMapWeight(w->layers[i].wgate, f, p->dim* p->hidden_dim, scratchCpu);
        memoryMapWeight(w->layers[i].wdown, f, p->hidden_dim* p->dim, scratchCpu);

        memoryMapWeight(w->layers[i].rms_att_weight, f, p->dim * sizeof(half), scratchCpu);
        memoryMapWeight(w->layers[i].rms_ffn_weight, f, p->dim * sizeof(half), scratchCpu);
    }
    printf("done!\n");
    free(scratchCpu);
    return 0;
}

// initialization: read from checkpoint
void memoryMapWeight(void* op, FILE* fp, size_t bytes, void* scratch) {
    if (fread(scratch, 1, bytes, fp) != bytes) { printf("error reading weights");  exit(EXIT_FAILURE); }
    cudaMemcpyAsync(op, scratch, bytes, cudaMemcpyHostToDevice);
}

// ----------------------------------------------------------------------------
// neural net blocks
cudaStream_t stream;
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
void RoPERotation(T *q, T *k, int num_heads, int num_kv_heads, int head_size, int* pPos, int loff, float rope_theta) {
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
    
    RoPERotation_kernel <<<num_heads, head_size / 2, 0, stream >>> (q, k, num_kv_heads, head_size, pPos, loff, rope_theta);
}

template <typename T>
void MultiHeadAttention(T *output, T *q, T *key_cache, T *value_cache, T *att, int num_heads, int head_size, int loff, int seq_len) {
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
     * @param   seq_len           current position in the sequence
     * 
     * @tparam  T                 Data type of device params
     */

    int dim = head_size * num_heads;
    MultiHeadAttention_kernel <<<num_heads, 1024>>> (output, q, key_cache, value_cache, att, num_heads, head_size, loff, seq_len, dim);
}

template <typename T>
__global__ void copy_embedding_kernel(T* x, const T* __restrict__ table, int size, int* tokens, int* pPos)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    int pos = *pPos;
    int token = tokens[pos];
    int table_index = index + token * size;
    x[index] = table[table_index];
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

__global__ void mat_vec_kernel(half* op, half* ip, half* wt, int n, int d, int numSerialLoads,
    int ip_stride, int w_stride, int op_stride, int w_row_stride, float alpha) {
    int index = blockIdx.x * blockDim.y + threadIdx.y;
    if (index >= d)
        return;
    const half* __restrict__ input = ip + blockIdx.y * ip_stride;
    const half* __restrict__ weight = wt + blockIdx.y * w_stride;
    half* output = op + blockIdx.y * op_stride;

    float sum = 0;

    for (int i = 0; i < numSerialLoads; i++) {
        int j = i * 32 + threadIdx.x;
        if (j < n) {
            (float)weight[index * w_row_stride + j]*(float)input[j];
        }
    }

    using WarpReduce = cub::WarpReduce<float>;
    __shared__ typename WarpReduce::TempStorage temp;
    sum = WarpReduce(temp).Sum(sum);
    sum *= alpha;

    if (threadIdx.x == 0)
        output[index] = (half)sum;
}

template <typename T>
__device__ float get_mat_vec(int index, T* __restrict__ input, T * __restrict__ weight, int inputElements, int opElements, int numSerialElements) {
    int n = inputElements;
    float sum = 0;
    for (int i = 0; i < numSerialElements; i++) {
        int j = i * 32 + threadIdx.x;
        if (j < n)
            sum += ((float)weight[index * n + j]) * ((float)input[j]);
    }
    using WarpReduce = cub::WarpReduce<float>;
    __shared__ typename WarpReduce::TempStorage temp;
    sum = WarpReduce(temp).Sum(sum);
    return sum;
}

template <typename T>
__device__ void mat_vec_kernel_optional_accum(T* output, T* input, T* weight, int inputElements, int opElements, int numSerialElements, bool accum, int loff, int* pPos){

    int index = blockIdx.x * blockDim.y + threadIdx.y;
    if (index >= opElements)
        return;
    float sum = get_mat_vec(index, input, weight, inputElements, opElements, numSerialElements);
    if (threadIdx.x == 0) {
        if (loff != -1) {
            output += loff + (*pPos * opElements);
        }

        if (accum)
            sum += (float)output[index];
        output[index] = (T)sum;
    }
}

template <typename T>
__global__ void mat_vec_kernel(T* output, T* input, T * weight, int inputElements, int opElements, int numSerialElements, bool accum, int loff, int* pPos){
        mat_vec_kernel_optional_accum(output, input, weight, inputElements, opElements, numSerialElements, accum, loff, pPos);
}



template <typename T>
__global__ void mat_vec_kernel_MHA(T* op, T* ip, T* wt, int n, int numSerialElements,
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
void matmul(T* xout, T* x, T* w, int n, int d, int batch = 1, int x_stride = 0, int w_stride = 0, int op_stride = 0, int w_row_stride = -1, float alpha = 1.0f) {
    if ((n & 7) || (d & 7)) { printf("\nUnsupported matmul size. Exiting\n"); exit(EXIT_FAILURE); }
    int serialElements = divUp(n, 32);
    int serialLoads = divUp(serialElements, 8);     // we load 8 elements in parallel
    dim3 block_dim(32, 4);
    dim3 grid_dim(divUp(d, 4), batch);
    if (w_row_stride == -1) w_row_stride = n;
    mat_vec_kernel <<<grid_dim, block_dim, 0, stream >>> (xout, x, w, n, d, serialLoads, x_stride, w_stride, op_stride, w_row_stride, alpha);
}

template <typename T>
void matmul(T* xout, T* x, T* w, int inpSize, int opSize, bool accum = false, int loff = -1, int *pPos = nullptr) {
    if ((inpSize & 7) || (opSize & 7)) { printf("\nUnsupported matmul size. Exiting\n"); exit(EXIT_FAILURE); }
    // We are assuming a vector - matrix mul with col major matrix: height = inpSize,  width  = opSize
    int numSerialElements = divUp(inpSize, 32);
    dim3 block_dim(32, 4);
    dim3 grid_dim(divUp(opSize, 4), 1);
    mat_vec_kernel <<<grid_dim, block_dim, 0, stream >>> (xout, x, w, inpSize, opSize, numSerialElements, accum, loff, pPos);
}



template <typename T>
void ffn_matvec_silu(T* xout, T* x, T* gate_w, T* up_w, int inpSize, int opSize) {
    if ((inpSize & 7) || (opSize & 7)) { printf("\nUnsupported matmul size. Exiting\n"); exit(EXIT_FAILURE); }
    // We are assuming a vector - matrix mul with col major matrix: height = inpSize,  width  = opSize
    int numSerialElements = divUp(inpSize, 32);
    dim3 block_dim(32, 4);
    dim3 grid_dim(divUp(opSize, 4), 1);
    ffn_matvec_silu_kernel <<<grid_dim, block_dim, 0, stream >>> (xout, x, gate_w, up_w, inpSize, opSize, numSerialElements);
}

template <typename T>
__global__ void  ffn_matvec_silu_kernel(T* __restrict__ output, T* __restrict__ input, T* gate_weight, T* up_weight, int inputElements, int opElements, int numSerialElements) {

    int index = blockIdx.x * blockDim.y + threadIdx.y;
    if (index >= opElements)
        return;
        // matmul(s->hb, s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim);
        // matmul(s->hb2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim);
    float gate_val = get_mat_vec(index, input, gate_weight, inputElements, opElements, numSerialElements);
    float up_val = get_mat_vec(index, input, up_weight, inputElements, opElements, numSerialElements);

    // apply silu and write the result
    if (threadIdx.x == 0) {
        float val = gate_val;
        val *= 1.0f / (1.0f + expf(-val));
        val *= up_val;
        output[index] = (T)val;
    }
}

template <typename T1, typename T2>
void transformer(int *pos, Config* p, RunState<T1, T2>* s, TransformerWeights<T1>* w, int fusedMHA){
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
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery

    copy_embedding_kernel <<<divUp(dim, 256), 256, 0, stream >>> (x, w->token_embedding_table, dim, s->shared_data->tokens, pos);

    // forward all the layers
    for (int l = 0; l < p->n_layers; l++) {
        // kv cache layer offset for convenience
        int loff = l * p->seq_len * kv_dim;
        // attention rmsnorm
        rmsnorm(s->xb, x, w->layers[l].rms_att_weight, dim);
        
        // qkv matmuls for this position
        matmul(s->q, s->xb, w->layers[l].wq, dim, dim, false);
        matmul(s->key_cache, s->xb, w->layers[l].wk, dim, kv_dim, false);
        matmul(s->value_cache, s->xb, w->layers[l].wv, dim, kv_dim, false);

        // apply RoPE rotation to the q and k vectors for each head
        RoPERotation(s->q, s->key_cache, p->n_heads, p->n_kv_heads, head_size, pos, loff, p->rope_theta);

        if (fusedMHA){ MultiHeadAttentionFused(s->xb, s->q, s->key_cache+loff, s->value_cache+loff, s->att, p->n_heads, head_size, kv_mul, p->seq_len, pos);}
        // else MultiHeadAttention(s->xb, s->q, s->key_cache, s->value_cache, p->n_heads, head_size, loff, pos+1);
//    MultiHeadAttention(s->xb, s->q, s->key_cache + loff, s->value_cache + loff, s->att, p->n_heads, head_size, kv_mul, seq_len_bin, pPos);
        // final matmul to get the output of the attention
        matmul(s->x, s->xb, w->layers[l].wo, dim, dim, true);

        // // residual connection back into x
        // accum(x, s->xb2, dim);

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->layers[l].rms_ffn_weight, dim);

        // for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))       
        // apply gate proj and up proj and then the silu activation in a single fused kernel
        ffn_matvec_silu(s->hb, s->xb, w->layers[l].wgate, w->layers[l].wup, dim, hidden_dim);
 
          // final matmul (down proj) to get the output of the ffn fused with residual connection back into x
        matmul(s->x, s->hb, w->layers[l].wdown, hidden_dim, dim, true);
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size, false);

    // copy logits from GPU->CPU and convert_fp16_to_fp32
    // convert<float> <<<divUp(p->vocab_size, 256), 256 >>> (s->logits_temp, s->logits, p->vocab_size);
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

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;

    // buffer only used with nucleus sampling
    cudaMalloc((void**) & sampler->indices, vocab_size * sizeof(int));
}

void free_sampler(Sampler* sampler) {
    cudaFree(sampler->indices);
    cudaFree(sampler->tempStorage_sort);
    cudaFree(sampler->tempStorage_scan);
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

char* decode(Tokenizer* t, int prev_token, int token) {
    char* piece = t->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == bos_token && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // bad byte, don't print it
        }
    }
    printf("%s", piece);
}

template <typename T>
__global__ void convert_fp16_to_fp32(float* out, T* in, int elements) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements)
        out[index] = (float)in[index];
}


template <typename T1, typename T2>
void run_transformer(bool gen_token, Config* config, RunState<T1, T2>* state, TransformerWeights<T1>* weights, bool copyLogits, Sampler *pSampler, int fusedMHA) {
    transformer(state->pos, config, state, weights, fusedMHA);
    if (copyLogits) {
        // copy to the right slot in logits_array (and convert to FP32)
        // we compute perplexity on the CPU later.
        float* pOutput = state->logits_array + config->vocab_size * state->shared_data->pos;
        convert_fp16_to_fp32 << < divUp(config->vocab_size, 128), 128, 0, stream >> > (pOutput, state->logits, config->vocab_size);
    }
    sample(pSampler, state, gen_token, stream);
}

template <typename T1, typename T2>
void generate_tokens(char *prompt, char* checkpoint, int steps, Tokenizer* tokenizer, Sampler* pSampler, Config* config, 
TransformerWeights<T1>* weights, RunState<T1, T2>* state, int shared_weights, int weight_quant_num, int fusedMHA, unsigned long rng_seed){
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
        prompt_tokens = (int*)malloc(config->seq_len * sizeof(int));
        bpe_encode(prompt, tokenizer->vocab, tokenizer->vocab_scores, config->vocab_size, tokenizer->max_token_length, prompt_tokens, &num_prompt_tokens);
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
        // wait for GPU work for previous iteration to complete
        // the idea is to keep GPU working in parallel with any CPU work (e.g, printing tokens to console).
        cudaStreamSynchronize(stream);
        // Perf note: don't put CPU work here "before" calling transformer as it won't overlap with GPU execution.
        run_transformer(pos >= num_prompt_tokens - 1, config, state, weights, false, pSampler, fusedMHA); // forward the transformer to get next token

        if (pos > 0) {
            next = state->shared_data->tokens[pos];  // Note: this is output token from previous iteration
            char* piece = decode(tokenizer, token, next);
            safe_printf(piece);             // same as printf("%s", piece), but skips "unsafe" bytes
            if (next == eos_token) break;   // break if EOS token is reached
            // advance forward
            token = next;
        }
        pos++;
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
    free_run_state(state);
    free_weights(weights, shared_weights);

    if (prompt_tokens != NULL) free(prompt_tokens);
    Sleep(1000);
}

int run_inference_fp16(char* checkpoint, char* prompt, float temperature, float topp, int steps, unsigned long rng_seed, int weight_quant_num, int fusedMHA){
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
    TransformerWeights<half> weights;
    RunState<half, float> state;
    Sampler sampler;
    FILE* file = load_checkpoint(checkpoint);
    char default_tokenizer_path[] = "tokenizer.bin";

    // read in the config header
    if (load_config(file, &config)) {printf("load config failed\n"); Sleep(1000); return 1;}
    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    int shared_weights = config.vocab_size > 0 ? 1 : 0;
    // build the Sampler    
    config.vocab_size = abs(config.vocab_size);
    build_sampler(&sampler, config.vocab_size, temperature, topp, rng_seed);

    // read in the Transformer weights
    if (load_weights(file, &config, &weights, shared_weights, weight_quant_num)) {printf("load weights failed \n"); Sleep(1000); return 1;}
    fclose(file);
 
    // right now we cannot run for more than config.seq_len steps
    if (steps <= 0 || steps > config.seq_len) { steps = config.seq_len; }
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, default_tokenizer_path, config.vocab_size);
    // create and init the application RunState
    malloc_run_state(&state, &config);
    generate_tokens(prompt, checkpoint, steps, &tokenizer, &sampler, &config, &weights, &state, shared_weights, weight_quant_num, fusedMHA, rng_seed);
   
    return 0;
}

int run_inference_fp32(char* checkpoint, char* prompt, float temperature, float topp, int steps, unsigned long rng_seed, int weight_quant_num, int fusedMHA){
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
    TransformerWeights<float> weights;
    RunState<float, float> state;
    Sampler sampler;
    FILE* file = load_checkpoint(checkpoint);
    char default_tokenizer_path[] = "tokenizer.bin";

    // read in the config header
    if (load_config(file, &config)) {printf("load config failed\n"); Sleep(1000); return 1;}
    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    int shared_weights = config.vocab_size > 0 ? 1 : 0;
    config.vocab_size = abs(config.vocab_size);
    build_sampler(&sampler, config.vocab_size, temperature, topp, rng_seed);
    // read in the Transformer weights
    if (load_weights(file, &config, &weights, shared_weights, weight_quant_num)) {printf("load weights failed \n"); Sleep(1000); return 1;}
    fclose(file);
 
    // right now we cannot run for more than config.seq_len steps
    if (steps <= 0 || steps > config.seq_len) { steps = config.seq_len; }

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, default_tokenizer_path, config.vocab_size);

    // create and init the application RunState
    malloc_run_state(&state, &config);
    generate_tokens(prompt, checkpoint, steps, &tokenizer, &sampler, &config, &weights, &state, shared_weights, weight_quant_num, fusedMHA, rng_seed);
    free_tokenizer(&tokenizer);
    return 0;
}

// // ----------------------------------------------------------------------------

int main(int argc, char *argv[]) {

    char *checkpoint = NULL;  // e.g. out/model.bin
    float temperature = 0.9f; // e.g. 1.0, or 0.0
    int steps = 256;          // max number of steps to run for, 0: use seq_len
    char *prompt = NULL;      // prompt string
    int weight_quant = 1;     // default using half data type for device params 
    int fusedMHA = 1;
    float topp = 0.6; 
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

    if (argc >= 7) {
        // optional fusedMHA for faster inference 
        fusedMHA = atoi(argv[6]);
    }
    // seed rng with time. if you want deterministic behavior use temperature 0.0
    unsigned long rng_seed;
    rng_seed = (unsigned int)time(NULL);

    if (topp < 0.0 || 1.0 < topp) topp = 0.6;
    // read in the model.bin file
    if (weight_quant == 1){
        printf("using half precision on device\n");
       
        // if (run_inference_fp16(checkpoint, prompt, temperature, topp, steps, rng_seed, weight_quant, fusedMHA)) return 1;
    } else{
        printf("using full precision on device\n");
       
        if (run_inference_fp32(checkpoint, prompt, temperature, topp, steps, rng_seed, weight_quant, fusedMHA)) return 1;
    }
    return 0;

}