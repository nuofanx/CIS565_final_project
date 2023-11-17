
#include <cstdio>
#include <fstream>
#include <iostream>
#include "src/run.cuh"
#include <cuda_runtime.h>


#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

int CEIL_DIV(int a, int size){
    return (a -1) / size +1; 
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Please select a kernel (range 0 - 8, 0 for NVIDIA cuBLAS)"
                << std::endl;
        exit(EXIT_FAILURE);
    }
    // get environment variable for device
    int deviceIdx = 0;
    if (getenv("DEVICE") != NULL) {
    deviceIdx = atoi(getenv("DEVICE"));
    }
    // get kernel number
    int kernel_num = std::stoi(argv[1]);
    if (kernel_num < 0 || kernel_num > 12) {
        std::cerr << "Please enter a valid kernel number (0-8)" << std::endl;
        exit(EXIT_FAILURE);
    }

    cudaCheck(cudaSetDevice(deviceIdx));

    printf("Running kernel %d on device %d.\n", kernel_num, deviceIdx);

    // init structs
    Config config;
    TransformerWeights weights;
    // read in the model.bin file
    FILE* file = fopen(checkpoint, "rb");
    if (!file) {
        //  printf(..) does not do any flushing itself, it's the buffering of stdout that may flush when seeing a newline (if it's line-buffered). 
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

    float* x = state->x;
    int dim = config->dim;
    int hidden_dim = config->hidden_dim;
    int head_size = dim / config->n_heads;
    int weight_quant_num = 0;
    // test matmul kernel 
    int l = 0;
    int max_l = config->n_layers;
    run_matmul_gpu(s->q, s->xb, w->wq + l * dim * dim, dim, dim, kernel_num, weight_quant_num);
    
    // memory cleanup
    free_run_state_gpu(&state);
    free_weights_gpu(&weights, shared_weights);
    for (int i = 0; i < config.vocab_size; i++) { free(vocab[i]); }
        free(vocab);
    return 0;
}