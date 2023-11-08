#ifndef RUN_H
#define RUN_H

// struct declaration 
struct RunState;
struct Config;
struct TransformerWeights;

// memory allocation/deallocation functions 
void malloc_run_state_gpu(RunState* s, Config* p);
void free_run_state_gpu(RunState* s);
void malloc_weights_gpu(TransformerWeights* w, Config* p, int shared_weights);
void free_weights_gpu(TransformerWeights* w, int shared_weights);
// model loading function
int memcpy(void *w, int elements, FILE* f, void *block_cpu, void *block_gpu);
int checkpoint_init_weights(TransformerWeights *w, Config* p, FILE* f, int shared_weights);
// neural net functions
void siLU_gpu(float* a, float* b, int size);
void accum_gpu(float *a, float *b, int size);
void rmsnorm_gpu(float* o, float* x, float* weight, int size);
void softmax_gpu(float* x, int size);
void matmul_gpu(float* xout, float* x, float* w, int n, int d);
void transformer(int token, int pos, Config* p, RunState* s, TransformerWeights* w);
// utility 
int sample(float* probabilities, int n);
int argmax(float* v, int n);
long time_in_ms();

#endif
