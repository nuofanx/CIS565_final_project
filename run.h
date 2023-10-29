#ifndef RUN_H
#define RUN_H

// memory allocation/deallocation 
void malloc_run_state(RunState* s, Config* p);
void free_run_state(RunState* s);
void malloc_weights(TransformerWeights* w, Config* p);
void free_weights(TransformerWeights* w, int shared_weights);
// model loading 
int checkpoint_init_weights(TransformerWeights *w, Config* p, FILE* f);
// neural net 
void accum(float *a, float *b, int size);
void rmsnorm(float* o, float* x, float* weight, int size);
void softmax(float* x, int size);
void matmul(float* xout, float* x, float* w, int n, int d);
void transformer(int token, int pos, Config* p, RunState* s, TransformerWeights* w);
// utility 
int sample(float* probabilities, int n);
int argmax(float* v, int n);

#endif
