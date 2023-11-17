#include <stdio.h>
#include <stdlib.h> 
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <math.h>
#include <string.h>
// Put the function declaration in some header file, say myfuncs.hpp, then put the function definitions in a cpp file, say myfuncs.cpp.
// Make sure myfuncs.cpp includes myfuncs.hpp and then for every other cpp file just include myfuncs.hpp.
#include "src/run.cuh"


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

// parallelization strategies and expereiments 
// https://hd10.dev/posts/my-interests-2/cs259.pdf


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


int CEIL_DIV(int a, int size){
    return (a -1) / size +1; 
}

// Access coalescing is done at kernel runtime by the hardware. This makes sense since coalescing requires aligned access, which cannot be guaranteed at compile time as we pass the matrix pointers as function arguments. 
// ***

void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};


// function that selects the desired matmul kernel and the calculation precision in gpu    
void run_matmul_gpu(void* C, void* A, void* B, int M, int N, int K, int kernel_num, int weight_quant_num) {
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


