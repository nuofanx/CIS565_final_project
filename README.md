# CIS565_final_project - CUDA implementation of llama2 with weight quantization 
# basic explanation for llama2.cpp 
https://github.com/RahulSChand/llama2.c-for-dummies
# basic explanation for matmul 
https://siboehm.com/articles/22/CUDA-MMM

# Train instructions 
Find a text dataset, for example, a very simple one  
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)

```bash
python tinystories.py download
python tinystories.py pretokenize
```

Then train the model:

```bash
python train.py
```

# Run instructions
1. Preprocess command: 
   Check if tokenizer.bin file exists, if not, download tokenizer.model by running 
   wget https://karpathy.ai/llama2c/model.bin -P out
2. Compile command: \
   on windows (): \
   gcc -O3 -o run run.cpp \ 
   on Mac: \
   clang++ -O3 -o run run.cpp  
                    
3. Run command: \
   Default cpu: \
   ./run.cpp \
   CUDA (windows only) \ 
   ./run.cpp -gpu

# TODO
1. Write run.cu to support parallelization
    - [x] define three structs Config, runStates, and transformerWeights, which contain dimension parameters, state parameters and weight parameters respectively 
    - [x] write GPU momory allocation and deallocation functions for each parameter in those structs
    - [x] write initialization function that support random init or checkpoint read
    - [x] write basic gpu version of transformer operations \
       i) rmsnorm \
       ii) sum/accumulation\
       iii) softmax \
       iv) matmul 
    - [x] write more efficient implementations of matmul operations
    - [ ] write more efficient implementations of multihead attention  
    - [x] write transformer archetecture \
       i) attention (rmsnorm & matmul) \
       ii) RoPE positional embeddings (product and minus for each attention head) \
       iii) multiquery attention (dot product & softmax) \
       iv) final matmul to get the output of the attention \
       v) residual connection back into x \
       vi) ffn rmsnorm and ffn (rmsnorm, matmul, sigmoid, elementwise multiply, final matmul) \
       vii) residual connection (accum(state.x, state.xb)) 
    - [x] write other utility functions \
       i) argmax \
       ii) sample
    - [x] write the main code to execute the program that supports the following: \
       i) random init/checkpoint reading (define gpu memory, load model onto gpu) \
       ii) model inference (write a loop to forward the transformer to get logits for the next token on gpu) \
       iii) report our achieved tok/s\
       iv) free gpu memory after the run 
     
2. write weight quantization code 


# Neural Net Operation Optimization that could be attempted 
1. matmul operation 
<!-- https://siboehm.com/articles/22/CUDA-MMM -->
- Blocktiling
- Vectorized SMEM and GMEM Access
- Warptiling

2. rotation embedding 
splits the input tensors xq and xk into real and imaginary parts, reshapes freqs_cos and freqs_sin for broadcasting, performs calculations analogous to complex multiplication, and finally combines the real and imaginary parts back into the original tensor.

3. RMS operation - Fused kernel 
<!-- https://ai.lefebvre-sarrut.eu/2023/07/20/deep-dive-into-kernel-fusion-accelerating-inference-in-llama-v2/#unleashing-enhanced-efficiency-simplified-fusions-in-rmsnorm-computation-with-triton -->

4. Softmax operation 
<!-- https://oneflow2020.medium.com/how-to-implement-an-efficient-softmax-cuda-kernel-oneflow-performance-optimization-sharing-405ad56e9031 -->
- Warp and block tiling 
A Warp processes one or two rows of computation for the case num_cols <= 1024.
A block processes one row of computation and uses Shared Memory to store the intermediate result data, for cases where the required Shared Memory resources meet the bootable condition of Kernel Launch, which in this test environment is 1024 < num_cols <= 4096. A Block processes a row of computation without using Shared Memory, and reads input x repeatedly, for cases where (1) and (2) are not supported.

- Pack Half types into Half2 for access, increasing instruction transfer without changing latency, similar to CUDA template for element-wise kernels optimization.

- Bank Conflicts in Shared Memory.

# Results 
1. Speed of CUDA implementation vs python implementation vs pure C++ implementation 
<!-- include table -->
Inference speed using implementations of different language  
| python | pure C++ | C++ and CUDA |
| ------------- | ------------- | ------------- |
| Content Cell  | Content Cell  | Content Cell |

2. Effect of matmul operation (gpu calculation speed improvment and analysis)
Calculations: 
Naive implementation
| ------------- | -------------  | 
| **Total FLOPS** |               = GFLOPS|
| **Total data to read (minimum!)**  |  = MB|
| **Total data to store**  |  = MB|

3. GPU specification 
| Metric | Value |
| ------------- | ------------- | 
| Compute Capability | Content Cell  | 
| max threads per block |  |
| max threads per multiprocessor | | 
| threads per warp | |
| warp allocation granularity | |

the resource demands for our kernel:
|Registers per Thread	| 37|
|SMEM per Block	|8192 B|
|Threads per Block|	1024|

Comparison 

|        | Naive implementation  |  Global Memory Coalescing| Shared Memory Blocking|  
| ------------- | ------------- | ------------- |
| **Memory throughput**| Content Cell  | Content Cell | Content Cell |
| **Performance** | Content Cell  | Content Cell | Content Cell |

|        | 1d blocktiling  |  2d blocktiling| Shared Memory Blocking|  
| ------------- | ------------- | ------------- |
| **results per thread** | Content Cell    |  
| **Performance** | Content Cell  | Content Cell | 
| **memory access**| Content Cell  | Content Cell | 



Performance of the best kernel so far against cuBLAS across increasing matrix size:

<!-- Include figure -->
<!-- ![best performing kernel vs cuBLAS] (/assets/images/electrocat.png) -->

3. Effect of multihead attention (speed improvement and anaylsis) 


4. Effect of weight quantization on inference quality vs speed

Inference speed comparison with/without Weight quantization
|                | **FP16** | **FP32**|
| **Naive**  | Content Cell    |  Content Cell |
| **Global Memory Coalescing** | Content Cell |Content Cell |
| **Shared Memory Blocking** | Content Cell |Content Cell |
| **Shared Memory Blocking** | Content Cell |Content Cell |
| **1d blocktiling** | Content Cell |Content Cell |
| **2d blocktiling** | Content Cell |Content Cell |
| **warptiling** | Content Cell |Content Cell |
| **cub reduce** | Content Cell  | Content Cell | 
| **cuBLAS** | Content Cell  | Content Cell | 

Inference result
Before:
After:
