#include <stdio.h>
#include <cuda_runtime.h>
#include "run.cuh" 
#include <cub/cub.cuh>
#include "kernels.cu"
#include <cublas_v2.h>

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

int CEIL_DIV(int a, int size){
    return (a -1) / size +1; 
}

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

//*** utility functions 
long time_in_ms(){
    struct timespec time;
    timespec_get(&time, TIME_UTC);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
 }

int argmax(float* v, int n) {
     // return argmax of v in elements 0..n
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

void softmax(float* x, int size) {
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

int sample(float* probabilities, int n) {
    // sample index from probabilities, they must sum to 1
    float r = (float)rand() / (float)RAND_MAX;
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (r < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

//***

// *** 
//   TODO: neural net functions 
void run_accum_gpu(float *a, float *b, int size){
    // call kernel function with block number CEIL_DIV(size, 256) and thread size 256
    elementwiseAddKernel <<< CEIL_DIV(size, 256), 256 >>> (a, b, size);
}

void run_rmsnorm_gpu(float* o, float* x, float* weight, int size){
    // calculate the blocks needed 
    int elementsPerThread = CEIL_DIV(size, 1024);
    // call the kernel with one single block and 1024 threads per block 
    rmsNormKernel<<<1,1024>>>(o, x, weight, size, elementsPerThread);
}

void run_RoPERotation_gpu(float *q, float *k, float *f_real, float *f_imag, int num_heads, int head_size) {
    RoPERotation_kernel <<<num_heads, head_size / 2 >>> (q, k, f_real, f_imag, num_heads, head_size);
}

//  Sigmoid Linear Unit (SiLU)  
void run_siluElementwiseMul_gpu(float * hb, float* hb2, int size){
    int threadNumber = 256;
    int blockNumber = CEIL_DIV(size, );
    siluElementwiseMulKernel<<blockNumber, threadNumber>> (hb, hb2, size);
}


// takes in token, pos, and model weights 
void transformer(int token, int pos, Config* p, RunState* s, TransformerWeights* w, int kernel_num, int weight_quant_num){
    float* x = s->x;
    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;
     // copy the token embedding into x
    float* content_row = &(w->token_embedding_table[token * dim]);
    cudaMemcpyAsync(x, content_row, dim * sizeof(float), cudaMemcpyDeviceToDevice);

    float* freq_cis_real_row = w->freq_cis_real + pos * head_size / 2;
    float* freq_cis_imag_row = w->freq_cis_imag + pos * head_size / 2;

    // forward all the layers
    for (int l = 0; l < p->n_layers; l++) {

        // attention rmsnorm
        run_rmsnorm_gpu(s->xb, x, w->rms_att_weight + l * dim, dim);

        // qkv matmuls for this position
        run_matmul_gpu(s->q, s->xb, w->wq + l * dim * dim, dim, dim, kernel_num, weight_quant_num);
        run_matmul_gpu(s->k, s->xb, w->wk + l * dim * dim, dim, dim, kernel_num, weight_quant_num);
        run_matmul_gpu(s->v, s->xb, w->wv + l * dim * dim, dim, dim, kernel_num, weight_quant_num);

        // apply RoPE rotation to the q and k vectors for each head
        run_RoPERotation_gpu(s->q, s->k, freq_cis_real_row, freq_cis_imag_row, p->n_heads, head_size);

         // save key,value at this time step (pos) to our kv cache
        int loff = l * p->seq_len * dim; // kv cache layer offset for convenience
        float * key_cache_row = s->key_cache + loff + pos * dim;
        float* value_cache_row = s->value_cache + loff + pos * dim;
        cudaMemcpyAsync(key_cache_row, s->k, dim * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(value_cache_row, s->v, dim * sizeof(float), cudaMemcpyDeviceToDevice);

        MultiHeadAttention(s->xb, s->q, s->key_cache, s->value_cache, p->n_heads, head_size, loff, pos+1);

        // final matmul to get the output of the attention
        run_matmul_gpu(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim, kernel_num, weight_quant_num);

        // residual connection back into x
        run_accum_gpu(x, s->xb2, dim);

        // ffn rmsnorm
        run_rmsnorm_gpu(s->xb, x, w->rms_ffn_weight + l * dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x) 
        // (dim, hidden_dim) x (hidden_dim, ) 
        run_matmul_gpu(s->hb, s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim, kernel_num, weight_quant_num);
        run_matmul_gpu(s->hb2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim, kernel_num, weight_quant_num);

        // apply F.silu activation on hb and multiply it with hb2
        run_siluElementwiseMul_gpu(s->hb, s->hb2, hidden_dim);

        // final matmul to get the output of the ffn
        run_matmul_gpu(s->xb, s->hb, w->w2 + l * dim * hidden_dim, hidden_dim, dim, kernel_num, weight_quant_num);

         // residual connection
        run_accum_gpu(x, s->xb, dim);
    }

    // final rmsnorm
    run_rmsnorm_gpu(x, x, w->rms_final_weight, dim);

    // classifier into logits
    run_matmul_gpu(s->logits_gpu, x, w->wcls, p->dim, p->vocab_size, kernel_num, weight_quant_num);

    switch (weight_quant){
        // does not do anything, float -> float    
        case 0: 
            break;
        // convert half to float 
        case 1:
            ConvertFP16toFP32 <<<CEIL_DIV(p->vocab_size, 256), 256 >>> (s->logits_temp, s->logits_gpu, p->vocab_size);
            break;
        default: 
            throw std::invalid_argument("Unknown weight quantization number");
    }
    
    // copy logits from GPU->CPU
    cudaMemcpy(s->logits, s->logits_temp, p->vocab_size * sizeof(float), cudaMemcpyDeviceToHost);
}

// checkpoint loading functions
int Memcpy(void *w, int elements, FILE* f, void *scratch_cpu, void *scratch_gpu, int weight_quant_num){
    int count = fread(scratch_cpu, sizeof(float), elements, f);
    if (count != elements) return 1; // report error by return 1 if not match
    cudaMemcpyAsync(scratch_gpu, scratch_cpu, sizeof(float) * elements, cudaMemcpyHostToDevice);
    switch (weight_quant_num){
        case 0:
            break;
        case 1:
            ConvertFP32toFP16 <<<divUp(elements, 256), 256 >>> ((half*)w, (float*)scratchGpu, elements);
            break;
        default:
            throw std::invalid_argument("Unknown weight quantization number");
    }
    return 0;
}

void checkpoint_init_weights(TransformerWeights *w, Config* p, FILE* f, int shared_weights, int weight_quant_num){
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

void run_matmul_gpu_naive(void *C, void* A, void*B, int M, int N, int K, int weight_quant_num){
    // calculates dot product of x and w, store in xout
    // n is the input dimension, d is the dimension
    // define 3D dimension parameters
    // 32 * 32 = 1024 thread per block
    dim3 blockDim(32, 32,1);
    dim3 gridDim(CEIL_DIV(M,32), CEIL_DIV(N,32));  
    // kernel for matmul operation
    matmulKernel_naive<<<gridDim, blockDim>>>(C, A, B, M, N, K, weight_quant_num);
}

void run_matmul_gpu_global_mem_coalesce(void *C, void* A, void*B, int M, int N, int K, int weight_quant_num){
    // calculates dot product of x and w, store in xout
    // n is the input dimension, d is the dimension
    // define 3D dimension parameters
    // 32 * 32 = 1024 thread per block
    dim3 blockDim(32, 32);
    // ceil(M,32) and ceil(N,32) blocks 
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N,32));
    // kernel for matmul operation
    matmulKernel_global_mem_coalesce<32>
        <<<gridDim, blockDim>>>(C, A, B, M, N, K, weight_quant_num);
}

void run_matmul_gpu_shared_mem_blocking(void *C, void* A, void*B, int M, int N, int K, int weight_quant_num){
    dim3 blockDim(32, 32);
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N,32));
    matmulKernel_shared_mem_blocking<<<gridDim, blockDim>>>(C, A, B, M, N, K, weight_quant_num);
}

void run_matmul_gpu_1d_blocktiling(void *C, void* A, void*B, int M, int N, int K, int weight_quant_num){
    dim3 blockDim(32, 32);
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N,32));
    
    matmulKernel_1d_blocktiling<<<gridDim, blockDim>>>(C, A, B, M, N, K, weight_quant_num);
}

void run_matmul_gpu_2d_blocktiling(void *C, void* A, void*B, int M, int N, int K, int weight_quant_num){
    // dim3 blockDim(32, 32);
    // dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N,32));
    const uint BK = 8;
    const uint TM = 8;
    const uint TN = 8;
    const uint BM = 128;
    const uint BN = 128;
    while (M < 128 and N < 128) {
        BM /= 2;
        BN /= 2;   
    }
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    matmulKernel_2d_blocktiling<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(C, A, B, M, N, K, weight_quant_num);
}

void run_matmul_gpu_vectorized(void *C, void* A, void*B, int M, int N, int K, int weight_quant_num){
    const uint BK = 8;
    const uint TM = 8;
    const uint TN = 8;
    while (M < 128 and N < 128) {
        BM /= 2;
        BN /= 2;   
    }
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    matmulKernel_2d_blocktiling<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(C, A, B, M, N, K, weight_quant_num);
}

void run_matmul_gpu_warptiling(void *C, void* A, void*B, int M, int N, int K, int weight_quant_num){
    if (check_parameter(BM, BN, BK, WM, WN, NUM_THREADS, TM, TN, WN_ITER, WN_ITER)) return;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim(NUM_THREADS);
    matmulKernel_warptiling<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS><<<gridDim, blockDim>>>(C, A, B, M, N, K);
}

void run_matmul_gpu_cubreduce(void *C, void* A, void*B, int M, int N, int K, int weight_quant_num){
    matmulKernel_cubreduce(C, A, B, M, N, K, 0);
}

void run_matmul_gpu_cublas(void *C, void* A, void*B, int M, int N, int K, int weight_quant_num){
    cudaError_t cudaStat;  // cudaMalloc status
    cublasStatus_t stat;   // cuBLAS functions status
    cublasHandle_t handle; // cuBLAS context
    stat = cublasCreate(&handle); // initialize CUBLAS context
    float alpha = 1.0f;
    float beta = 0.0f;
    switch (weight_quant_num){
        case 0:
            runCublasFP32(handle, M, N, K, alpha, A, B, beta, C);
            break;
        case 1:
            runCublasFP16(handle, M, N, K, alpha, A, B, beta, C);
            break;
        default:
            throw std::invalid_argument("Unknown weight quantization number");
        }
    cublasDestroy(handle);
}


void checkpoint_init_weights(TransformerWeights *w, Config* p, FILE* f, int shared_weights, int weight_quant_num){
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


// function that selects the desired matmul kernel and the calculation precision in gpu    
void run_matmul_gpu(void* C, void* A, void* B, int M, int N, int kernel_num, int weight_quant_num) {
    // check if the input matrices are squared matrices, if so, apply more efficiemt implementaions 
    if (!(M == N && N==K)){
        kernel_num = 8;
        printf('non square matrices encountered in matrix multiplication'); 
        fflush(stdout);   
    } 
    switch (kernel_num) {
        case 0:
            run_matmul_gpu_cublas(C,A,B, M, N, K);
            break;
        case 1:
            run_matmul_naive(C, A, B, M, N, K);
            break;
        case 2:
            run_matmul_gpu_global_mem_coalesce(C, A, B, M, N, K);
            break;
        case 3:
            run_matmul_gpu_shared_mem_block(C, A, B, M, N, K);
            break;
        case 4:
            run_matmul_gpu_1d_blocktiling(C, A, B, M, N, K);
            break;
        case 5:
            run_matmul_gpu_2d_blocktiling(C, A, B, M, N, K);
            break;
        case 6:
            run_matmul_gpu_vectorized(C, A, B, M, N, K);
            break;
        case 7:
            run_matmul_gpu_warptiling(C, A, B, M, N, K);
            break;
        default:
            throw std::invalid_argument("Unknown kernel number");
    }
}