
## CIS565 Final Project - Running LLama2 fast inference on CUDA
Name: Nuofan Xu 

## Device 
| Type | Name|
|------|-----|
|GPU   | NVIDIA RTX 2080 Super |
|CPU   | AMD 3700              |

# Introduction

The Llama2 model stands out as a pivotal advancement in natural language processing due to its transformative impact on fast and accurate inference tasks. Renowned for its superior architecture and refined training methods, Llama2 achieves remarkable performance across various language-based tasks. Its significance lies in its ability to enable rapid inference, crucial in real-time applications like conversational AI, chatbots, and voice assistants. By efficiently processing and understanding complex language structures, Llama2 ensures swift and accurate responses, enhancing user experience and enabling seamless interactions in today's fast-paced digital landscape. Its speed and precision in inference contribute significantly to improved efficiency and effectiveness in numerous practical applications, making Llama2 a cornerstone model in the realm of natural language processing.


Adapted from the llama2.c repository, this codebase offers a CUDA-powered inference solution designed for the Llama2 LLM. The primary emphasis revolves around simplicity and minimalism. Furthermore, it delves into assessing the speed of diverse implementations against Nvidia's CuBLAS kernel. For demonstration and ease, only petite models trained by karapathy, the author of llama2.cpp, find usage in this project. As detailed within the llama2.cpp repository, remarkably petite LLMs exhibit commendable performance, especially when the domain is tailored narrowly (reference: TinyStories paper). Notably, it's worth mentioning that, as the neural net architecture is identical, one can follow the prescribed steps by karapathy to convert Meta's 7B and larger models into .bin format, facilitating inference operations seamlessly.


## Setup

1. Install dependencies: CUDA toolkit 12, Python (+ Seaborn).
2. Download pretrained models from huggingface hub [tinyllamas](https://huggingface.co/karpathy/tinyllamas) in the llama2.cu format .bin from below:

## Pretrained models
| model | dim | n_layers | n_heads | max context length | parameters | val loss | download
| --- | --- | --- | --- | --- | --- | --- | --- |
| 15M | 288 | 6 | 6 | 256 | 15M | 1.072 | [stories15M.bin](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin) |
| 42M| 512 | 8 | 8 | 1024 | 42M | 0.847 | [stories42M.bin](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin) |
| 110M| 768 | 12 | 12 | 1024 | 110M | 0.760 | [stories110M.bin](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin) |


## Complie & Run Commands
For ease of use, all the essential code for each task is consolidated within a single file. Users only need to run a compile command without the necessity of creating complex CMakeLists.txt files. Subsequently, they can run the generated executable file effortlessly.

| task                      | complie command                 | run command                |
|---------------------------|---------------------------------|----------------------------|
|verify matvec kernel       |nvcc gemv_test.cu -o build/gemv  |start build/sgemv_test.exe  |
|matvec kernel benchmarking |nvcc sgemv.cu -o build/sgemv     |start build/sgemv.exe       | 
|matmul kernel benchmarking |nvcc sgemm.cu -o build/sgemm     |start build/sgemm.exe       |
|MHA benchmarking           |nvcc mha_compare.cu -o build/mha_compare     |start build/mha_compare.exe       |
|llama2 inference with CUDA |nvcc llama2.cu -o build/llama2   |start build/llama2.exe "MODEL_NAME"    |

where MODEL_NAME is any of ["stories15M.bin", "stories52M.bin", "stories110M.bin"]

-compliation tips:\
*if Nividia cublas libarary is not linked properly, add [-lcublas] \
*if the compiler complains about operators on half daat type, add -arch=sm_XX where XX is the compute capability of your GPU. If the error persists, it's likely that some features used in this project is not supported on this device.\
*if path to build tool is not found, add path manually with [-ccbin "PATH_TO_VS_BUILD_TOOL"] 

Example: \
nvcc gemv_verify.cu -o build/verify [-ccbin "PATH_TO_VS_BUILD_TOOL"] [-lcublas]\
nvcc llama2.cu -o build/llama2 \
start llama2.cu "stories52M.bin"

## Results generation commands
Once the executable files are generated and executed, users can utilize the corresponding Python file to collect information and generate result tables and figures.

| task                                  | run command                     |
|---------------------------------------|---------------------------------|
|generate matmul benchmarking result    | python gen_sgemm_results.py     |
|generate matvecmul benchmarking result | python gen_sgemv_results.py     |
|generate mha benchmarking result       | python gen_mha_results.py       |
|generate llama2 inference benchmarking | python gen_inference_results.py |
*results are generated and updated in the README.md file.

## matrix vector multiplication benchmarking result:
![](sgemv_benchmark_results.png)

<!-- sgemv_benchmark_results -->
| Kernel              |   mflops/s | Performance relative to cuBLAS   |
|:--------------------|-----------:|:---------------------------------|
| 2: cuBLAS(F)        |    3039.9  | 99.8%                            |
| 0: cuBLAS(C)        |    3045.7  | 100.0%                           |
| 3: SMEM Chaching(F) |    7331.1  | 240.7%                           |
| 1: SMEM Caching(C)  |    8551.08 | 280.8%                           |
<!-- sgemv_benchmark_results -->
*(C stands for row major order and F stands for column major order)


The weight matrices are read in row-major order. Kernels were implemented and compared in both row-major and column-major orders. The analysis revealed that row-major order exhibits superior speed in both the CuBLAS and SMEM implementations.

In terms of matrix vector multiplication, it is surprising to see that, the CuBLAS library does not provide the best performance. The custom implementation with shared memory coalescing and warp level optimization provides better efficiency.  

## matrix matrix multiplication benchmarkign result:
![](sgemm_benchmark_results.png)

GFLOPs at matrix size 4096x4096:
<!-- sgemm_benchmark_results -->
| Kernel                              |   GFLOPs/s | Performance relative to cuBLAS   |
|:------------------------------------|-----------:|:---------------------------------|
| 1: Naive                            |      46.05 | 0.9%                             |
| 2: GMEM Coalescing                  |     613.02 | 12.0%                            |
| 3: SMEM Caching                     |     810.03 | 15.8%                            |
| 4: 1D Blocktiling                   |    1966.05 | 38.4%                            |
| 8: Avoid Bank Conflicts (Offset)    |    3165.48 | 61.9%                            |
| 5: 2D Blocktiling                   |    3493.09 | 68.3%                            |
| 6: Vectorized Mem Access            |    3747.18 | 73.2%                            |
| 7: Avoid Bank Conflicts (Linearize) |    4192    | 81.9%                            |
| 9: Warptiling                       |    4646.98 | 90.8%                            |
| 0: cuBLAS                           |    5116.1  | 100.0%                           |
<!-- sgemm_benchmark_results -->

In terms of matrix matrix multiplication, we can see that, cuBLAS library provides superior performance in terms of GFLOP/s. With all the optimization techniques used, the custom matmul kernel only achieves about 91% speed of its CuBLAS counterpart.

# MHA implementation fusion vs vertical parallelization 
![](fusion_benchmark_results.png)
<!-- fusion_benchmark_results -->
| method   |   dim |   time |
|:---------|------:|-------:|
| fusion   |   288 |   1.54 |
| fusion   |  1024 |   4    |
| fusion   |   512 |   3    |
| vertical |   288 |   1.3  |
| vertical |  1024 |   3.8  |
| vertical |   512 |   2.5  |
<!-- fusion_benchmark_results -->


## inference benchmarking result 
![](inference_benchmark_results.png)
<!-- inference_benchmark_results -->
| device   | model    |    speed |   dim |   hidden_dim |   n_layers |   n_heads |   n_kv_heads |   kflops | params   |
|:---------|:---------|---------:|------:|-------------:|-----------:|----------:|-------------:|---------:|:---------|
| cpu      | cpu_110M |   45.12  |   768 |         2048 |         12 |        12 |           12 | 3145.73  | 110M     |
| cpu      | cpu_15M  |  250.21  |   288 |          768 |          6 |         6 |            6 |  442.368 | 15M      |
| cpu      | cpu_42M  |  120.91  |   512 |         1376 |          8 |         8 |            8 | 1409.02  | 42M      |
| gpu      | gpu_110M |  580.866 |   768 |         2048 |         12 |        12 |           12 | 3145.73  | 110M     |
| gpu      | gpu_15M  | 1040.82  |   288 |          768 |          6 |         6 |            6 |  442.368 | 15M      |
| gpu      | gpu_42M  |  850     |   512 |         1376 |          8 |         8 |            8 | 1409.02  | 42M      |
<!-- inference_benchmark_results -->

It can be clearly observed that CUDA implementaion provides a performance boost compared to its CPU counterparts. However, the token generateion speed of CUDA implementation seems to decrease faster with the increase of the arithemic complexity of mat vec multiplication. 

## Examples of generated tokens 
<!-- inference_token_generation_results -->
| model   | tokens                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
|:--------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| gpu110M | Once upon a time, there was a little girl named Lily. She loved to play with her toys and run around outside. One day, her mommy asked her to help with the laundry. Lily didn't want to help because she wanted to play, but her mommy said it was important to help out.Lily started to help with the laundry, but she got bored quickly. She started to play with the clothes and throw them around. Her mommy saw what she was doing and got upset. She told Lily that it was not nice to play with the laundry and that she needed to help her.Lily felt bad and knew she had done something wrong. She promised her mommy that she would help with the laundry from now on. Her mommy forgave her and they finished the laundry together. Lily learned that it's important to help out and that being kind is always the right thing to do.<s>Once upon a time, there was a little girl named Lily. She loved to play with her toys and run around outside. One day, her mommy asked her to help with the laundry. Lily didn't want                                                                     |
| gpu15M  | Once upon a time, there was a young boy named Tim. Tim loved to sing. Every day, he would sing happy songs. His favorite song was about the earth. He liked to watch the earth grow.One day, Tim saw a big bird. The bird could sing too! Tim asked the bird, "Can you teach me how to sing like you?" The bird said, "Yes, I can teach you. Just keep singing." So, Tim started to learn about the earth. He sang every day.After many days, Tim played with the bird. They sang together. One day, they heard a loud noise. They looked up and saw a big storm coming. The storm had hit the earth. Tim was sad. But then, something unexpected happened. The bird sang a beautiful song, and they saw that the earth was safe. Tim was happy, and they sang together until the storm went away.<s>Once upon a time, there was a young boy named Tim. Tim loved to build things with his toy blocks. One day, Tim's mom gave him a big calendar. She said, "Tim, one day, you will build something big or strong."Tim was very excited. He wanted to build something big.vocab_size：(32000)seq_length: 256 |
| gpu42M  | Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big wall that scared her. She wanted to run away, but she wanted to be brave. So, she took a deep breath and walked towards the wall. Suddenly, she heard a loud noise and everything became quiet. She looked up and saw a bird flying over the wall. It was so pretty! Lily felt happy and proud of herself for being brave. From that day on, she wasn't scared of walls anymore.<s>Once upon a time, there was a little girl named Lily. She loved to play outside and explore the world around her. One day, she went on a walk with her mommy and they saw a big gray cloud in the sky.Lily asked her mommy, "What is that big gray thing in the sky?"Her mommy replied, "That's a cloud, sweetie. It means it might rain soon."Lily looked up at the cloud and said, "I hope it doesn't rain on us!"Her mommy smiled and said, "Don't worry, Lily. We                                                                                                                                  |
<!-- inference_token_generation_results -->




## Conclusion:
The CUDA implementation of llama2.c indeed provides a remarkable performance enhancement in terms of speed. The token generation speed surged from approximately 250 tokens per second to 1100 tokens per second using an RTX 2080 Super GPU. Further analysis of the forward pass of the transformer model highlights that the significant speed boost primarily stems from the efficient implementation of the matvec/matmat multiplication kernel and the adept parallelization of the multihead attention mechanism.

In assessing the matmul kernel, seven distinct implementations were developed: naive, global memory coalescing, shared memory coalescing, 1D block tiling, 2D block tiling, warptiling, etc., for comprehensive comparison. Additionally, Nvidia's cuBLAS library was used as the baseline. Results indicate that cuBLAS delivers the most efficient implementation of general matrix-matrix multiplication, while the most efficient custom implementation achieves close to 90% efficacy. cuBLAS gains an advantage due to its comprehensive support for all input shapes, albeit at a cost - it houses not just one, but hundreds of implementations of SGEMM, making the cuBLAS library a hefty 500MB of compiled code.

Regarding the matvec kernel, two custom kernels supporting column and row major orders were developed. These were compared with the CuBLAS implementation and demonstrated superior efficiency. The evaluation revealed that the row-major order aligns naturally with the data storage format, rendering it faster, and therefore, it was adopted in this project.

Concerning the parallelization of multihead attention, column parallelization, also known as vertical parallelization, emerged as the preferred scheme. This method proves advantageous as it eliminates the need for inter-node communication when nodes possess replicated weights. Hence, this parallelization scheme was selected for implementation. While an alternative version of multihead attention parallelization exists, it wasn't implemented in this project for direct comparison purposes.

## Area of improvments:
1. Further study on more efficient implementation of matmul and matvecmul kernel  
    - For example, double buffering, for better interleaving of computation and memory loading. For now, see CUTLASS Pipelining. In CUTLASS, double buffering is done on two levels: GMEM ⇒ SMEM, and SMEM ⇒ Registerfile.

2. More efficient implementation of multihead attention 
    - better usage of kernel fusion - kernel fusion was attemped in llama2_fused.cu but the code has not yet been fully debugged. The main problem is that cub library seem to not provide full support for operations in half data type, and consequently sorting and picking the top p probabiltiies in sampling becomes challenging. The fused multihead attention kernel and other kernels were successfully implemented.  

3. weight quantization for less memory and computing requirement 
    - Currently: float16 are used for storage but calculations are done in full precision float32. Although it might cause minimal loss in accuracy, compared to float32, it reduces model size by up to half. As the architecture is identical, you can also load and inference Meta's Llama 2 models. However, the current code only inferences models in fp32, so you will most likely not be able to productively load models larger than 7B. 
    - potentially useful quantization weight schemes 
        - float 16 weight quantization 
            - It supports some delegates (e.g. the GPU delegate) which can operate directly on float16 data, resulting in faster execution than float32 computations.
            - It does not reduce latency as much as a quantization to fixed point math.
        - int8 quantization
            - get further latency improvements, reductions in peak memory usage, and compatibility with integer only hardware devices or accelerators by making sure all model math is integer quantized.
            - need to calibrate or estimate the range, i.e, (min, max) of all floating-point tensors in the model. Unlike constant tensors such as weights and biases, variable tensors such as model input, activations (outputs of intermediate layers) and model output cannot be calibrated unless we run a few inference cycles.As a result, the converter requires a representative dataset to calibrate them. 
        - 16x8 quantization
            - The main advantage of this quantization is that it can improve accuracy significantly, but only slightly increase model size. 
            - activations are quantized based on their range to 16-bits, weights are quantized in 8-bit integer and bias is quantized into 64-bit integer. 
            - Currently inference is noticeably slower than 8-bit full integer due to the lack of optimized kernel implementation.
            - Currently it is incompatible with the existing hardware accelerated TFLite delegates. 


# Disclaimer 
Some code that are not gpu parallelizable are directly adopted from llama2.cpp repository as there is no need to reinvent the wheel. Ideas of how to efficiently parallelize the CUDA kernels used in this project are referenced below.

# References:
1. Multihead attention parallelization 
column row-parallelization vs column parallelization 
https://insujang.github.io/2022-08-03/analyzing-parallelization-of-attention/

2. Parallelization schemes of multihead attention: cpu naive vs gpu horizontal / gpu vertical
https://hd10.dev/posts/my-interests-2/cs259.pdf

3. General Matrix Matrix multiplication 
https://siboehm.com/articles/22/CUDA-MMM

4. Matrix Vector Multiplication 
https://github.com/uysalere/cuda-matrix-vector-multiplication/tree/master

5. Softmax kernel
https://oneflow2020.medium.com/how-to-implement-an-efficient-softmax-cuda-kernel-oneflow-performance-optimization-sharing-405ad56e9031

## Credit
1. [wangzyon/NVIDIA_SGEMM_PRACTICE](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE) for the general matrix matrix multiplication and matrix vector multiplication benchmarking setup.
2. [uysalere/cuda-matrix-vector-multiplication](https://github.com/uysalere/cuda-matrix-vector-multiplication/blob/master/vmp.pdf) for the matrix vector multiplication implemenation ideas.
3. [karpathy/llama2.c](https://github.com/karpathy/llama2.c/) for the Llama2 inference implementation work flow and pretrained models for benchmarking. 
4. [ankan-ban/llama_cu_awq](https://github.com/ankan-ban/llama_cu_awq) for the Llama2 CUDA inference implementation work flow and weight quantization application.