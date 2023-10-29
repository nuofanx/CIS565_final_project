# CIS565_final_project
# Run instructions
1. Compile command:
2. Run command: 

# Changes:
1. Modifed readme file to show the work to be done 
<!--- 2. Wrote function interface for functions required
3. Finished struct definitions --->

# TODO
1. Write run.cu to support parallelization 
    - [ ] define three structs Config, runStates, and transformerWeights, which contain dimension parameters, state parameters and weight parameters respectively 
    - [ ] write GPU momory allocation and deallocation functions for each parameter in those structs
    - [ ] write initialization function that support random init or checkpoint read
    - [ ] write gpu version of transformer operations \
       i) rmsnorm \
       ii) sum/accumulation\
       iii) softmax \
       iv) matmul 
    - [ ] write transformer archetecture \
       i) attention (rmsnorm & matmul) \
       ii) RoPE positional embeddings (product and minus for each attention head) \
       iii) multiquery attention (dot product & softmax) \
       iv) final matmul to get the output of the attention \
       v) residual connection back into x \
       vi) ffn rmsnorm and ffn (rmsnorm, matmul, sigmoid, elementwise multiply, final matmul) \
       vii) residual connection (accum(state.x, state.xb)) 
    - [ ] write other utility functions \
       i) argmax \
       ii) sample
    - [ ] write the main code to execute the program that supports the following: \
       i) random init/checkpoint reading (define gpu memory, load model onto gpu) \
       ii) model inference (write a loop to forward the transformer to get logits for the next token on gpu) \
       iii) report our achieved tok/s\
       iv) free gpu memory after the run 
     
2. write weight quantization code 
