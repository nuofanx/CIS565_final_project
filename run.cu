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
void run_accum_gpu(float *a, float *b, int size){
    // call kernel function with block number CEIL_DIV(size, 256) and thread size 256
    elementwiseAddKernel <<< CEIL_DIV(size, 256), 256 >>> (a, b, size);
}


//  Sigmoid Linear Unit (SiLU)  
void run_siluElementwiseMul_gpu(float * hb, float* hb2, int size){
    int threadNumber = 256;
    int blockNumber = CEIL_DIV(size, );
    siluElementwiseMulKernel<<blockNumber, threadNumber>> (hb, hb2, size);
}

__global__ void multiHeadAttentionKernel_GEMM(){
    
}
// parallelization strategies and expereiments 
// https://hd10.dev/posts/my-interests-2/cs259.pdf
__global__ void multiHeadAttentionKernel_horizontal(){

}

void runMultiHeadAttention(float *output, float *q, float *key_cache, float *value_cache, int num_heads, int head_size, int loff, int seq_len) {
    int dim = head_size * num_heads;
    MultiHeadAttentionKernel_naive <<<num_heads, 1024>>> (output, q, key_cache, value_cache, num_heads, head_size, loff, seq_len, dim);
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

__global__ void run_softmax_gpu(float* x, int size){
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


int CEIL_DIV(int a, int size){
    return (a -1) / size +1; 
}

void run_matmul_gpu_naive(float* xout, float* x, float* w, int n, int d){
    // calculates dot product of x and w, store in xout
    // n is the input dimension, d is the dimension
    // define 3D dimension parameters
    // 32 * 32 = 1024 thread per block
    dim3 blockDim(32, 32,1);
    dim3 gridDim(CEIL_DIV(n,32), CEIL_DIV(d,32));
    // kernel for matmul operation
    matmulKernel_naive<<<gridDim, blockDim>>>(xout, x, w, n, d, 4);
    
}

void run_matmul_gpu_global_mem_coalesce(float* xout, float* x, float* w, int n, int d)
{
    // calculates dot product of x and w, store in xout
    // n is the input dimension, d is the dimension
    // define 3D dimension parameters
    // 32 * 32 = 1024 thread per block
    dim3 blockDim(32, 32);
    dim3 gridDim(CEIL_DIV(n, 32), CEIL_DIV(d,32));
    // kernel for matmul operation
    matmulKernel_global_mem_coalesce<<<gridDim, blockDim>>>(xout, x, w, n, d);
}

void run_matmul_gpu_shared_mem_blocking(float* xout, float* x, float* w, int n, int d){
    dim3 blockDim(32, 32);
    dim3 gridDim(CEIL_DIV(n, 32), CEIL_DIV(d,32));
    matmulKernel_shared_mem_blocking<<<gridDim, blockDim>>>(xout, x, w, n, d, k);
}

void run_matmul_gpu_1d_blocktiling(float* xout, float* x, float* w, int n, int d){
    dim3 blockDim(32, 32);
    dim3 gridDim(CEIL_DIV(n, 32), CEIL_DIV(d,32));
    matmulKernel_1d_blocktiling<<<gridDim, blockDim>>>(xout, x, w, n, d, k);
}



// Access coalescing is done at kernel runtime by the hardware. This makes sense since coalescing requires aligned access, which cannot be guaranteed at compile time as we pass the matrix pointers as function arguments. 
// ***

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

void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};

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




// takes in token, pos, and model weights 
void transformer(int token, int pos, Config* p, RunState* s, TransformerWeights* w, int kernel_num, int weight_quant){
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
    run_matmul_gpu(s->logits_gpu, x, w->wcls, p->dim, p->vocab_size);

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

// function that selects the calculation precision in gpu 
void run_matmul_gpu(float* output, float* input, float* weight, int input_dim, int output_dim, int hidden_dim, int kernel_num, int weight_quant_num){
    run_sgemm_matmul_kernel(output, input, weight, input_dim, output_dim, hidden_dim, kernel_num, weight_quant_num);
}
   
void run_sgemm_matmul_kernel(void* C, void* A, void* B, int M, int N, int K, int kernel_num, int weight_quant_num) {
    switch (kernel_num) {
        case 0:
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
            break;
        case 1:
            run_matmul_naive(C, A, B, M, N, K);
            break;
        case 2:
            run_matmul_gpu_global_mem_coalesce(C, A, B, M, N, K);
            break;
        case 3:
            run_matmul_shared_mem_block(C, A, B, M, N, K);
            break;
        case 4:
            run_matmul_1d_blocktiling(C, A, B, M, N, K);
            break;
        case 5:
            run_matmul_2d_blocktiling(C, A, B, M, N, K);
            break;
        case 6:
            run_matmul_vectorized(C, A, B, M, N, K);
            break;
        case 7:
            run_matmul_warptiling(C, A, B, M, N, K);
            break;
        default:
            throw std::invalid_argument("Unknown kernel number");
    }
}

int main(char* checkpoint){
    // define model parameters

    char *checkpoint = NULL;  // e.g. out/model.bin
    float temperature = 0.9f; // e.g. 1.0, or 0.0
    int steps = 256;          // max number of steps to run for, 0: use seq_len
    int weight_quant_num = 0; // default running gpu calc with 32fp
    int kernel_num = 1;       // default using naive kernel 
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
    // option of different implementation of matmul kernel 
    if (argc >=6){
        kernel_num = atoi(argv[5]);
        if (kernel_num < 0 || kernel_num > 12) {
            std::cerr << "Please enter a valid kernel number (0-12)" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    // option of different gpu calclulation precision 
    if (argc >= 7){
        weight_quant_num = atoi(argv[6]); 
    }
    // ***
    //   TODO: support cpu and gpu

    // ***

	// seed rng with time. if you want deterministic behavior use temperature 0.0
    srand((unsigned int)time(NULL)); 
  
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
        transformer(token, pos, &config, &state, &weights, kernel_num, weight_quant_num);

        // sample the next token
        if(temperature == 0.0f) {
            // greedy argmax sampling
            next = argmax(state.logits, config.vocab_size);
        } else {
            // apply the temperature to the logits
            for (int q=0; q<config.vocab_size; q++) {
                state.logits[q] /= temperature; 
            }
            // apply softmax to the logits to get the probabilities for next token
            run_softmax_gpu(state.logits, config.vocab_size);
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


