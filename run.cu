#include <stdio.h>
#include <run.h> 
#include <stdlib.h> 

#include <cuda_runtime.h>

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

}

void rmsnorm_gpu(float* o, float* x, float* weight, int size){

}

void softmax_gpu(float* x, int size){

}

__global__ void matmulKernel_naive(float* output, float* input, float* weight, int n, int d, int numElements){
    // use the grid, block and thread hierarchy to assign each thread a unique entry in the result matrix C. 
    // compute position in new matrix that this thread is responsible for
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    // threadIdx varies from 0 to 31, blockIdx varies from CEIL_DIV(n,32) to CEIL_DIV(d,32) 
    // loop through each element that needs to be computed 
    if (x< n && y < d){
        float tmp = 0.0f;

        for (int i = 0; i < numElements; i++) {
            // indexing into strided in-memory representations of matrices.
            tmp += input[x * numElements + i] * weight[i* d + y];

        }
        output[x*numElements + y] = tmp;
    }
}

__global__ void matmulKernel_gmc(float* output, float* input, float* weight, int n, int d, int numElements){
    // use the grid, block and thread hierarchy to assign each thread a unique entry in the result matrix C. 
    // compute position in new matrix that this thread is responsible for
    const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);
    // threadIdx varies from 0 to 31, blockIdx varies from CEIL_DIV(n,32) to CEIL_DIV(d,32) 
    // loop through each element that needs to be computed 
    if (x< n && y < d){
        float tmp = 0.0f;

        for (int i = 0; i < numElements; i++) {
            // indexing into strided in-memory representations of matrices.
            tmp += input[x * numElements + i] * weight[i* d + y];

        }
        output[x*numElements + y] = tmp;
    }
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

//Shared Memory Cache-Blocking
void matmul_gpu_smc(float* xout, float* x, float* w, int n, int d){
    const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);
    // advance pointers to the starting positions
    A += cRow * BLOCKSIZE * K;                    // row=cRow, col=0
    B += cCol * BLOCKSIZE;                        // row=0, col=cCol
    C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // row=cRow, col=cCol

    float tmp = 0.0;
    // the outer loop advances A along the columns and B along
    // the rows until we have fully calculated the result in C.
    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    // Have each thread load one of the elements in A & B from
    // global memory into shared memory.
    // Make the threadCol (=threadIdx.x) the consecutive index
    // to allow global memory access coalescing
    As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

    // block threads in this block until cache is fully populated
    __syncthreads();

    // advance pointers onto next chunk
    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

    // execute the dotproduct on the currently cached block
    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
        tmp += As[threadRow * BLOCKSIZE + dotIdx] *
                Bs[dotIdx * BLOCKSIZE + threadCol];
    }
    // need to sync again at the end, to avoid faster threads
    // fetching the next block into the cache before slower threads are done
    __syncthreads();
    }
    C[threadRow * N + threadCol] =
        alpha * tmp + beta * C[threadRow * N + threadCol];
}



// we explicitly cached the entry of B into Btmp and reordered the two inner loops for efficiency.

// allocate thread-local cache for results in registerfile
float threadResults[TM] = {0.0};

// outer loop over block tiles
for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
  // populate the SMEM caches (same as before)
  As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
  Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
  __syncthreads();

  // advance blocktile for outer loop
  A += BK;
  B += BK * N;

  // calculate per-thread results
  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    // we make the dotproduct loop the outside loop, which facilitates
    // reuse of the Bs entry, which we can cache in a tmp var.
    float Btmp = Bs[dotIdx * BN + threadCol];
    for (uint resIdx = 0; resIdx < TM; ++resIdx) {
      threadResults[resIdx] +=
          As[(threadRow * TM + resIdx) * BK + dotIdx] * Btmp;
    }
  }
  __syncthreads();
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

// Increasing Arithmetic Intensity via 2D Blocktiling
__global__ void matmulKernel_2dBlocktiling(){
    // compute a grid of 8*8 elements of C per thread. 
    // The first stage of the kernel is for all threads to work together to populate the SMEM cache. 
    for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
    As[(innerRowA + loadOffset) * BK + innerColA] =
        A[(innerRowA + loadOffset) * K + innerColA];
    }
    for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
    Bs[(innerRowB + loadOffset) * BN + innerColB] =
        B[(innerRowB + loadOffset) * N + innerColB];
    }

    __syncthreads();


    // 
    // allocate thread-local cache for results in registerfile
    float threadResults[TM * TN] = {0.0};
    // register caches for As and Bs
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    // outer-most loop over block tiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
        As[(innerRowA + loadOffset) * BK + innerColA] =
            A[(innerRowA + loadOffset) * K + innerColA];
    }
    for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
        Bs[(innerRowB + loadOffset) * BN + innerColB] =
            B[(innerRowB + loadOffset) * N + innerColB];
    }
    __syncthreads();

    // advance blocktile
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
        // load relevant As & Bs entries into registers
        for (uint i = 0; i < TM; ++i) {
        regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
        }
        for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
        }
        // perform outer product on register cache, accumulate
        // into threadResults
        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            threadResults[resIdxM * TN + resIdxN] +=
                regM[resIdxM] * regN[resIdxN];
        }
        }
    }
    __syncthreads();
    }
}

// Vectorize SMEM and GMEM Accesses
__global__ void matmulKernel_vecSMEM_GMEM(){
    float4 tmp =
        reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
    // transpose A during the GMEM to SMEM transfer
    As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

    //why faster than just manually unrolling the access (or using pragma unroll)
    reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] =
        reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];
    __syncthreads();

    // the compiler has no way to verify that the float* B pointer that is passed to the kernel is 128b aligned, which would be a requirement for using LDG.E.128. So the reinterpret_cast’s only purpose is to promise the compiler that the float* B pointer will be aligned.
    // Bs[innerRowB * BN + innerColB * 4 + 0] = B[innerRowB * N + innerColB * 4 + 0];
    // Bs[innerRowB * BN + innerColB * 4 + 1] = B[innerRowB * N + innerColB * 4 + 1];
    // Bs[innerRowB * BN + innerColB * 4 + 2] = B[innerRowB * N + innerColB * 4 + 2];
    // Bs[innerRowB * BN + innerColB * 4 + 3] = B[innerRowB * N + innerColB * 4 + 3];
}


// Warptiling
// Blocktiling: Different blocks can execute in parallel on different SMs.
// Warptiling: Different warps can execute in parallel on different warp schedulers, and concurrently on the same warp scheduler.
// Threadtiling: (a very limited amount of) instructions can execute in parallel on the same CUDA cores (= instruction-level parallelism aka ILP).
// dotIdx loops over contents of SMEM

__global__ void matmulKernel_Warptiling(){
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
        // populate registers for this thread's part of the warptile
        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
            for (uint i = 0; i < TM; ++i) {
            regM[wSubRowIdx * TM + i] =
                As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
                    threadRowInWarp * TM + i];
            }
        }
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            for (uint i = 0; i < TN; ++i) {
            regN[wSubColIdx * TN + i] =
                Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
                    threadColInWarp * TN + i];
            }
        }
    }

    // execute warptile matmul. Later this will map well to
    // warp-wide matrix instructions, executed on tensor cores.
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            // calculate per-thread results with register-cache locality
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                                (wSubColIdx * TN) + resIdxN] +=
                    regM[wSubRowIdx * TM + resIdxM] *
                    regN[wSubColIdx * TN + resIdxN];
                }
            }
        }
    }
}

//***
//  TODO: timing functions
long time_in_ms();

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


