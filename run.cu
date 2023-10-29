#include <stdio.h>
#include <run.h> 
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

void malloc_run_state(RunState* s, Config* p) {
   cudaMalloc((void**)&s->x, p->dim * sizeof(float));
     cudaMalloc((void**)&s->xb, p->dim * sizeof(float));
     cudaMalloc((void**)&s->xb2, p->dim * sizeof(float));
     cudaMalloc((void**)&s->hb, p->hidden_dim * sizeof(float));
     cudaMalloc((void**)&s->hb2, p->hidden_dim * sizeof(float));
     cudaMalloc((void**)&s->q, p->dim * sizeof(float));
     cudaMalloc((void**)&s->k, p->dim * sizeof(float));
     cudaMalloc((void**)&s->v, p->dim * sizeof(float));
     cudaMalloc((void**)&s->logits_gpu, p->vocab_size * sizeof(half));
     cudaMalloc((void**)&s->key_cache, p->n_layers * p->seq_len * p->dim * sizeof(float));    // potentially huge allocs
     cudaMalloc((void**)&s->value_cache, p->n_layers * p->seq_len * p->dim * sizeof(float)
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q 
     || !s->k || !s->v || !s->att || !s->logits || !s->key_cache 
     || !s->value_cache) {
        printf("cuda malloc failed!\n");
        exit(1);
    }

void free_run_state(RunState* s) {
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

void malloc_weights(TransformerWeights* w, Config* p) {
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
  
    // ensure all mallocs went fine
    if (!w->token_embedding_table || !w->rms_att_weight || !w->rms_ffn_weight 
     || !w->wq || !w->wk || !w->wv || !w->wo || !w->w1 || !w->w2 || !w->w3 || 
        !w->rms_final_weight || !w->freq_cis_real || !w->freq_cis_imag) {
        printf("gpu malloc failed!\n");
        exit(1);
    }
}

void free_weights(TransformerWeights* w) {
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
 }



