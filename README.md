# CIS565_final_project - CUDA implementation of llama2 with weight quantization 
# basic explanation for llama2.cpp 
https://github.com/RahulSChand/llama2.c-for-dummies
# basic explanation for matmul 
https://siboehm.com/articles/22/CUDA-MMM

# Run instructions
1. Compile command: \
   on windows (): \
   gcc -O3 -o run run.cpp \ 
   on Mac: \
   clang++ -O3 -o run run.cpp  
                    
2. Run command: \
   Default cpu: \
   ./run.cpp \
   CUDA (windows only) \ 
   ./run.cpp -gpu

# Changes (until Oct.28):
1. Modifed readme file to show the work to be done 
2. Wrote function interface for functions required in run.h
3. Finished struct definitions and memory allocation/deallocation functions in run.cu

# Changes (until Oct.30):
1. Added main code (done main structure to generate prompt, still need to add support for tokenizer loading, cpu/gpu running, timing)
2. Wrote model and config loading code
3. Added TODOs in run.cu for next time (neural net operations, and finishing up main function) 
4. Slightly modified run.h to add struct declaration and shared weight argument 
5. Added run instructions (not working at the moment) 

# Changes (until Oct.31):
1. Added various matmul kernel methods that runs on GPU. The methods include: 
   a. Naive implementation
   b. 1D Blocktiling (for Calculating Multiple Results per Thread)
   c. 2D Blocktiling (for increasing Arithmetic Intensity)
   d. Vectorize SMEM and GMEM Accesses (using vectorized SMEM loads)
   e. Warptiling (add another hierarchy of tiling)
   The code is not finished, as this project will first focus on getting the code running and then implement speed improvements. The rest will be implemented if time is allowed.


# TODO
1. Write run.cu to support parallelization
    - [x] define three structs Config, runStates, and transformerWeights, which contain dimension parameters, state parameters and weight parameters respectively 
    - [x] write GPU momory allocation and deallocation functions for each parameter in those structs
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
