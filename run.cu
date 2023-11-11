#include <stdio.h>
#include <run.h> 
#include <stdlib.h> 
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include "kernels.cu"

// define Transformer weight, RunState and config structs

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct {
    // token embedding table
    float* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls
    float* wq; // (layer, dim, dim)
    float* wk; // (layer, dim, dim)
    float* wv; // (layer, dim, dim)
    float* wo; // (layer, dim, dim)
    // weights for ffn
    float* w1; // (layer, hidden_dim, dim)
    float* w2; // (layer, dim, hidden_dim)
    float* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // freq_cis for RoPE relatively positional embeddings
    float* freq_cis_real; // (seq_len, dim/2)
    float* freq_cis_imag; // (seq_len, dim/2)
} TransformerWeights;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (seq_len,)
    float *logits_gpu; // output logits gpu
    float *logits; // output logits cpu
    // kv cache
    float *key_cache;   // (layer, seq_len, dim)
    float *value_cache; // (layer, seq_len, dim)
} RunState;

void malloc_run_state_gpu(RunState* s, Config* p) {
     cudaMalloc((void**)&s->x, p->dim * sizeof(float));
     cudaMalloc((void**)&s->xb, p->dim * sizeof(float));
     cudaMalloc((void**)&s->xb2, p->dim * sizeof(float));
     cudaMalloc((void**)&s->hb, p->hidden_dim * sizeof(float));
     cudaMalloc((void**)&s->hb2, p->hidden_dim * sizeof(float));
     cudaMalloc((void**)&s->q, p->dim * sizeof(float));
     cudaMalloc((void**)&s->k, p->dim * sizeof(float));
     cudaMalloc((void**)&s->v, p->dim * sizeof(float));
     cudaMalloc((void**)&s->logits_gpu, p->vocab_size * sizeof(float));
     cudaMalloc((void**)&s->key_cache, p->n_layers * p->seq_len * p->dim * sizeof(float));    // potentially huge allocs
     cudaMalloc((void**)&s->value_cache, p->n_layers * p->seq_len * p->dim * sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q 
     || !s->k || !s->v || !s->att || !s->logits || !s->key_cache 
     || !s->value_cache) {
        printf("cuda malloc failed!\n");
        exit(1);
    }
}

void free_run_state_gpu(RunState* s) {
    cudaFree(s->x);
    cudaFree(s->xb);
    cudaFree(s->xb2);
    cudaFree(s->hb);
    cudaFree(s->hb2);
    cudaFree(s->q);
    cudaFree(s->k);
    cudaFree(s->v);
    cudaFree(s->att);
    cudaFree(s->logits_gpu);
    cudaFree(s->key_cache);
    cudaFree(s->value_cache);
    free(s->logits); // free logit cpu as well
}

void malloc_weights_gpu(TransformerWeights* w, Config* p, int shared_weights) {
    cudaMalloc((void**)&w->token_embedding_table, p->vocab_size * p->dim * sizeof(float));
    cudaMalloc((void**)&w->rms_att_weight, p->n_layers * p->dim * sizeof(float));
    cudaMalloc((void**)&w->rms_ffn_weight, p->n_layers * p->dim * sizeof(float));
    cudaMalloc((void**)&w->wq, p->n_layers * p->dim * p->dim * sizeof(float));
    cudaMalloc((void**)&w->wk, p->n_layers * p->dim * p->dim * sizeof(float));
    cudaMalloc((void**)&w->wv, p->n_layers * p->dim * p->dim * sizeof(float));
    cudaMalloc((void**)&w->wo, p->n_layers * p->dim * p->dim * sizeof(float));
    cudaMalloc((void**)&w->w1, p->n_layers * p->hidden_dim * p->dim * sizeof(float));
    cudaMalloc((void**)&w->w2, p->n_layers * p->dim * p->hidden_dim * sizeof(float));
    cudaMalloc((void**)&w->w3, p->n_layers * p->hidden_dim * p->dim * sizeof(float));
    cudaMalloc((void**)&w->rms_final_weight, p->dim * sizeof(float));
    cudaMalloc((void**)&w->freq_cis_real, p->seq_len * head_size / 2 * sizeof(float));
    cudaMalloc((void**)&w->freq_cis_imag, p->seq_len * head_size / 2 * sizeof(float));
    
    // use the same address if shared, or allocate new memory on GPU for wcls
    if (shared_weights)
        w->wcls = w->token_embedding_table;
    else
        cudaMalloc((void**)&w->wcls, p->vocab_size * p->dim * sizeof(float));

    // ensure all mallocs went fine
    if (!w->token_embedding_table || !w->rms_att_weight || !w->rms_ffn_weight 
     || !w->wq || !w->wk || !w->wv || !w->wo || !w->w1 || !w->w2 || !w->w3 || 
        !w->rms_final_weight || !w->freq_cis_real || !w->freq_cis_imag || !w->wcls) {
        printf("gpu malloc failed!\n");
        exit(1);
    }
}

void free_weights_gpu(TransformerWeights* w, int shared_weights) {
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
    // only free weights if weights are not shared 
    if (!shared_weights)
        cudaFree(w->wcls);
}

// *** 
//   TODO: neural net functions 
void accum_gpu(float *a, float *b, int size){
    // call kernel function with block number CEIL_DIV(size, 256) and thread size 256
    elementwiseAdd <<< CEIL_DIV(size, 256), 256 >>> (a, b, size);
}

__global__ void elementwiseAdd(float* dest, half* src, int size) {
     int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
     if (thread_index < size)
        dest[i] = dest[i] + src[i];
}

//  Sigmoid Linear Unit (SiLU)  
void siLU_gpu(float * hb, float* hb2, int size){
    int threadNumber = 256;
    int blockNumber = CEIL_DIV(size, );
    siluKernel<<blockNumber, threadNumber>> (hb, hb2, size);
}

// kernel function for x * sigma(x)
__global__ void siluKernel(float* dest, float* src, int size){
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_index < size){
        // extract val
        val = dest[i]
        // sigma(x)
        val *= 1.0f / (1.0f + expf(-val));
        // x * sigma(x)
        val *= src[i];
    }
}

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



__global__ void multiHeadAttentionKernel_GEMM(){
    
}
// parallelization strategies and expereiments 
// https://hd10.dev/posts/my-interests-2/cs259.pdf
__global__ void multiHeadAttentionKernel_horizontal(){

}

__global__ void multiHeadAttentionKernel_horizontal(){

}

__
// An important point to bear in mind is that when two consecutive load operations are carried out on the same addresses, the second operation is likely to get the data from the GPU cache, meaning it doesn't cost double DDR loading to perform two passes within the same Triton program.
// For instance, the reduction operation (sum) is executed outside the loop due to its cost (this CUDA presentation: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf will give you a basic understanding of how complicated reductions are at warp level).
// single block 
__global__ void rmsNormKernel(float* o, float* x, float* weight, int size){
    //  first loop over input tensor to compute the root mean of the square
    float ss = 0.0f;
    for (int i = 0; i < elementsPerThread; i++) {
        int index = threadIdx.x + i * 1024;
        if (index < size)
            ss += (float) x[index];
    }

    // calculate result and load to shared memory 
    __shared__ float shared_ss;
    // take mean, add eps, then sqrt 
    if (threadIdx.x == 0) {
        ss /= size;
        ss += 1e-5f;
        ss = 1.0f / sqrtf(ss);
        shared_ss = ss;
    }
    __syncthreads();
    // read from shared memory 
    ss = shared_ss;
    //  we keep this reduction operation outside the loop for perf reasons
    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage temp;
    ss = BlockReduce(temp).Sum(ss * ss);

    //  apply the normalization and multiply by RMS weights
    for (int i = 0; i < elementsPerThread; i++) {
        int index = threadIdx.x + i * 1024;
        if (index < size) {
            float val = (float)x[index];
            val *= ss * (float)weight[index];
            o[index] = (half)val;
        }
    }
}

void rmsnorm_gpu(float* o, float* x, float* weight, int size){
    // calculate the blocks needed 
    int elementsPerThread = CEIL_DIV(size, 1024);
    // call the kernel with one single block and 1024 threads per block 
    rmsNormKernel<<<1,1024>>>(o, x, weight, size, elementsPerThread);
}


void softmax_gpu(float* x, int size){
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


int CEIL_DIV(int a, int size){
    return (a -1) / size +1; 
}

void matmul_gpu_naive(float* xout, float* x, float* w, int n, int d){
    // calculates dot product of x and w, store in xout
    // n is the input dimension, d is the dimension
    // define 3D dimension parameters
    // 32 * 32 = 1024 thread per block
    dim3 blockDim(32, 32,1);
    dim3 gridDim(CEIL_DIV(n,32), CEIL_DIV(d,32));
    // kernel for matmul operation
    matmulKernel_naive<<<gridDim, blockDim>>>(xout, x, w, n, d, 4);
    
}

void matmul_gpu_gmc(float* xout, float* x, float* w, int n, int d)
{
    // calculates dot product of x and w, store in xout
    // n is the input dimension, d is the dimension
    // define 3D dimension parameters
    // 32 * 32 = 1024 thread per block
    dim3 blockDim(32, 32);
    dim3 gridDim(CEIL_DIV(n, 32), CEIL_DIV(d,32));
    // kernel for matmul operation
    matmulKernel_gmc<<<gridDim, blockDim>>>(xout, x, w, n, d);
}



void matmul_gpu_smcb(float* xout, float* x, float* w, int n, int d){
    dim3 blockDim(32, 32);
    dim3 gridDim(CEIL_DIV(n, 32), CEIL_DIV(d,32));
    matmulKernel_SMCB<<<gridDim, blockDim>>>(xout, x, w, n, d, k);
}

void matmul_gpu_1d_blocktiling(float* xout, float* x, float* w, int n, int d){
    dim3 blockDim(32, 32);
    dim3 gridDim(CEIL_DIV(n, 32), CEIL_DIV(d,32));
    matmulKernel_1d_blocktiling<<<gridDim, blockDim>>>(xout, x, w, n, d, k);
}



// Access coalescing is done at kernel runtime by the hardware. This makes sense since coalescing requires aligned access, which cannot be guaranteed at compile time as we pass the matrix pointers as function arguments. 
// ***

// checkpoint loading functions
int Memcpy(void *w, int elements, FILE* f, void *block_cpu, void *block_gpu){
    int count = fread(block_cpu, sizeof(float), elements, f);
    if (count != elements) return 1; // report error by return 1 if not match
    cudaMemcpyAsync(block_gpu, block_cpu, sizeof(float) * elements, cudaMemcpyHostToDevice);
    return 0;
}

void checkpoint_init_weights(TransformerWeights *w, Config* p, float* f, int shared_weights){
    // define the generic size that could contain all params 
    size_t largest_possible_size =std::max((size_t)p->vocab_size, p->n_layers * std::max(p->dim, p->hidden_dim))* p->dim * sizeof(float);
    // copy to gpu
    void* block_cpu = malloc(largest_possible_size);
    void* block_gpu = nullptr;
    cudaMalloc(&block_gpu, largest_possible_size);
    if (memcpy(w->token_embedding_table, p->vocab_size * p->dim, f, block_cpu, block_gpu)) return 1; 
    if (memcpy(w->rms_att_weight, p->n_layers * p->dim, f, block_cpu, block_gpu)) return 1;
    if (memcpy(w->wq, p->n_layers * p->dim * p->dim, f, block_cpu, block_gpu)) return 1;
    if (memcpy(w->wk, p->n_layers * p->dim * p->dim, f, block_cpu, block_gpu)) return 1;
    if (memcpy(w->wv, p->n_layers * p->dim * p->dim, f, block_cpu, block_gpu)) return 1;
    if (memcpy(w->wo, p->n_layers * p->dim * p->dim, f, block_cpu, block_gpu)) return 1;
    if (memcpy(w->rms_ffn_weight, p->n_layers * p->dim, f, block_cpu, block_gpu)) return 1;
     if (memcpy(w->w1, p->n_layers * p->dim * p->hidden_dim, f, block_cpu, block_gpu)) return 1;
     if (memcpy(w->w2, p->n_layers * p->hidden_dim * p->dim, f, block_cpu, block_gpu)) return 1;
     if (memcpy(w->w3, p->n_layers * p->dim * p->hidden_dim, f, block_cpu, block_gpu)) return 1;
     if (memcpy(w->rms_final_weight, p->dim, f, block_cpu, block_gpu)) return 1;

     int head_size = p->dim / p->n_heads;
     if (memcpy(w->freq_cis_real, p->seq_len * head_size / 2, f, block_cpu, block_gpu)) return 1;
     if (memcpy(w->freq_cis_imag, p->seq_len * head_size / 2, f, block_cpu, block_gpu)) return 1;

     if (!shared_weights)
         if (memcpy(w->wcls, p->vocab_size * p->dim, f, block_cpu, block_gpu)) return 1;

     cudaFree(block_gpu);
     free(block_cpu);
     return 0;
}


//***
long time_in_ms(){
    struct timespec time;
    timespec_get(&time, TIME_UTC);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
 }
//***

int main(char* checkpoint){
    // define model parameters

    char *checkpoint = NULL;  // e.g. out/model.bin
    float temperature = 0.9f; // e.g. 1.0, or 0.0
    int steps = 256;          // max number of steps to run for, 0: use seq_len
    //  // argparse, 'checkpoint' is the necessary arg
    if (argc < 2) {
        printf("Usage: %s <checkpoint_file> [temperature] [steps]\n", argv[0]);
        return 1;
    }
    if (argc >= 2) {
        checkpoint = argv[1];
    }
    if (argc >= 3) {
        // optional temperature. 0.0 = (deterministic) argmax sampling. 1.0 = baseline
        temperature = atof(argv[2]);
    }
    if (argc >= 4) {
        steps = atoi(argv[3]);
    }
    // option of running with cpu or gpu
    if (argc >=5) {
        use_gpu = atoi(argv[4]); 
    }
    // ***
    //   TODO: support cpu and gpu

    // ***

	// seed rng with time. if you want deterministic behavior use temperature 0.0
    srand((unsigned int)time(NULL)); 
    // read in the model.bin file
    Config config;
    TransformerWeights weights;
    // init structs
    Config config;
    TransformerWeights weights;
    // read in the model.bin file
    FILE* file = fopen(checkpoint, "rb");
    if (!file) {
        printf("Unable to open the checkpoint file %s!\n", checkpoint);
        return 1;
    }
    // read in the config header
    if (fread(&config, sizeof(Config), 1, file) != 1) { return 1; }
    // negative vocab size signaling unshared weights.
    int shared_weights = config.vocab_size > 0 ? 1 : 0;
    config.vocab_size = abs(config.vocab_size);
    
    // load model checkpoint 
    checkpoint_init_weights(&weights, &config, file);
    // allocate memory for Transformer weights
    malloc_weights_gpu(&weights, &config);
   
    // read in the tokenizer.bin file
    char** vocab = (char**)malloc(config.vocab_size * sizeof(char*));

    // ***
    //    TODO: read in the tokenizer.bin file 
    
    FILE* file = fopen("tokenizer.bin", "rb");
    if (!file) {
        // Run python tokenizer.py to convert tokenizer.model -> tokenizer.bin\n");
        printf("Unable to open tokenizer.bin!")
        return 1;
    }
    int len;
    for (int i = 0; i < config.vocab_size; i++) {
        // return if len cannot be read
        if (fread(&len, sizeof(int), 1, file) != 1) { return 1; }
        // allocate cpu memory for each char 
        vocab[i] = (char*)malloc(len + 1);
        // return if len does not match 
        if (fread(vocab[i], len, 1, file) != 1) { return 1; }
        vocab[i][len] = '\0'; // add the string terminating token
    }
    fclose(file);
    
    // *** 

    // create and init RunState struct
    RunState state;
    malloc_run_state_gpu(&state, &config);

    // the current position we are in
    long start = time_in_ms();

    int next;
    int token = 1; // 1 = BOS token in Llama-2 sentencepiece
    int pos = 0;
    while (pos < config.seq_len) {
         // forward the transformer to get logits for the next token
        transformer(token, pos, &config, &state, &weights);

        // sample the next token
        if(temperature == 0.0f) {
            // greedy argmax sampling
            next = argmax_gpu(state.logits, config.vocab_size);
        } else {
            // apply the temperature to the logits
            for (int q=0; q<config.vocab_size; q++) { state.logits[q] /= temperature; }
            // apply softmax to the logits to get the probabilities for next token
            softmax_gpu(state.logits, config.vocab_size);
            // we now want to sample from this distribution to get the next token
            next = sample(state.logits, config.vocab_size);
        }
        printf("%s", vocab[next]);
        fflush(stdout);

        // advance forward
        token = next;
        pos++;
    }
  
    // report achieved tok/s
    long end = time_in_ms();
    double time = (end - start) / 1000.0;
    printf("\nachieved tok/s: %f. Tokens: %d, seconds: %g\n", pos / time, pos, time);

    // memory cleanup
    free_run_state_gpu(&state);
    free_weights_gpu(&weights, shared_weights);
    for (int i = 0; i < config.vocab_size; i++) { free(vocab[i]); }
    free(vocab);
    return 0;
   
}


