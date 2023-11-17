#ifndef RUN_H
#define RUN_H

#include <stdio.h>

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
int Memcpy(void *w, int elements, FILE* f, void *scratch_cpu, void *scratch_gpu, int weight_quant_num);
int checkpoint_init_weights(TransformerWeights *w, Config* p, FILE* f, int shared_weights, int weight_quant_num);
// neural net functions
void run_siluElementwiseMul_gpu(float* a, float* b, int size);
void run_accum_gpu(float *a, float *b, int size);
void run_rmsnorm_gpu(float* o, float* x, float* weight, int size);
void run_softmax_gpu(float* x, int size);
void run_matmul_gpu(float* xout, float* x, float* w, int n, int d, int kernel_num, int weight_quant_num);
void softmax(float* x, int size);
void transformer(int token, int pos, Config* p, RunState* s, TransformerWeights* w, int kernel_num, int weight_quant_num);
// utility 
int sample(float* probabilities, int n);
int argmax(float* v, int n);
long time_in_ms();


#endif
