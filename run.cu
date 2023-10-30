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
void accum_gpu(float *a, float *b, int size);
void rmsnorm_gpu(float* o, float* x, float* weight, int size);
void softmax_gpu(float* x, int size);
void matmul_gpu(float* xout, float* x, float* w, int n, int d);
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
//  TODO: timing functions
long time_in_ms();

//

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


