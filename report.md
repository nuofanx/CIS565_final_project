# table of information for the kernels used in transformer inference 
dim, hidden_dim is defined by model configuration and is thus model dependent  

| Operation     | Description  | Block Number | Thread Number | Element per thread | info |
| --------------|--------------|--------------|---------------|--------------|--------------|
| **rmsnorm1**  | attention rmsnorm |  Ceil(m,1024)| 1024 | dont care | m = dim, using 1024 threads per block |
| **matmul1**   | Q = xb*wq  | Ceil(n,4) | (32, 4) | Ceil(m, 32) | m = n = dim, one output per warp, float32 | 
| **matmul2**   | K = xb*wk  | Ceil(n,4) | (32, 4) | Ceil(m, 32)| 
| **matmul3**   | V = xb*wv  | Ceil(n,4) | (32, 4) | Ceil(m, 32)| 
| **RoPERotation** | apply rotation |  num_heads |  head_size / 2  | 1 | Each block processes a single head
| **multiheadAttention** | Content Cell |num_heads | 1024 | dont care | Each block processes a single head | 
| **matmul4** | output of attention | Ceil(n,4) | (32, 4) | Ceil(m, 32) | | 
| **accum** | $x=x\oplus xb2$ | Ceil(m,256) | 256 | dont care | m = dim| 
| **rmsnorm** | ffn rmsnorm |  Ceil(m,1024)| 1024 | dont care | m = dim |  
| **matmul5** | hb=xb*w1 | Ceil(n,4) | (32, 4) | Ceil(m, 32) | m=dim, n=hidden_dim|
| **matmul6** | hb2=xb*w3 | Ceil(n,4) | (32, 4) | Ceil(m, 32) | m=dim, n=hidden_dim|
| **siluElementwiseMul** | hb = silu(hb,hb2)  | Content Cell | Content Cell | 
| **matmul7** | xb=hb*w2 | Ceil(hidden_dim, 256) | 256 | dont care | | 
| **accum** | x = x \oplus xb | 256 | 256| dont care| m = dim | 


## Test Results
Block Size of Tile | Matrix Dimension | GPU time in ms | CPU time in ms
--- | --- | :---:|:---:
8x8   | 16x16 & 16x16         |0.133    |0.0100
8x8   | 512x512 & 512x512     |0.295    |489.160
8x8   | 1024x1024 & 1024x1024 |2.299    |3896.926
16x16 | 16x16 & 16x16         |0.123    |0.0100
16x16 | 512x512 & 512x512     | 0.256 |465.544
16x16 | 1024x1024 & 1024x1024 | 1.966       |3885.956
32x32 | 16x16 & 16x16         | 0.143|0.0700
32x32 | 512x512 & 512x512     | 0.254       | 489.861
32x32 | 1024x1024 & 1024x1024 | 1.987|4085.987

default params
|               |  checkpoint 1  |  checkpoint 2 | checkpoint 3 | info         |
|---------------|----------------|---------------|--------------|--------------|
|**model_size** |  15MB          | 44MB          |              | |
|**dim**        |  288           | 512           | 256          | |
|**n_layers**   |  6             | 8             | 6            | |
|**n_heads**    |  6             | 8             | 6            | |
|**max_seq_len**|  256           | 1024          | 256          | |
|**BM**         |  128           | 128           | 128             |(BM,BK) is the chunk size of matrix A cached in SMEM
|**BN**         |  128           | 128           | 128| (BK,BN) is the chunk size of matrix B cached in SMEM
|**BK**         |  8             |  8            | 8  |(BM,BN) is the chunk size of matrix C
|**TM**         |  8             |  8            | 8  |(TM, TN) is the grid size in C that each thread has to calculate
|**TN**         |  8             |  8            | 8 |
|**WM**         |  4             |  4            | 4 | Each warp computes a chunk of size (WSUBN * WNITER) x (WSUBM * WMITER)
|**WN**         |  4             |  4            | 4 | Each thread computes WNITER * WMITER many chunks of size TM*TN.
|**WMITER**     |  4             |  4            | 4 | WSUBM = WM / WMITER;
|**WNITER**     |  4             |  4            | 4 | WSUBN = WN / WNITER; 

Parameter check:
BN % WN must be 0 
BM % WM must be 0
(BN / WN) * (BM / WM) == NUM_WARPS == 32
(WM * WN) % (WARPSIZE * TM * TN * WN_ITER) must be 0.
WM % WM_ITER must be 0 
WN % WN_ITER must be 0.
(NUM_THREADS * 4) % BK must be 0.

BM, BN and BK, which specify how much data we cache from GMEM into SMEM.
TM and TN, which specify how much data we cache from SMEM into the registers.
these were set to BM=BN=128 and BK=TM=TN=8. I wrote a bash script that searches through all sensible combinations and benchmarks their runtime.
depending on the GPU model

# Define the range of values for each parameter
| parameter name | list to autotune | 
|BK |(8 16 32 64)|
|BM |(64 128 256)|
|BN |(64 128 256)|
|WM |(32 64 128 256)|
|WN | (32 64 128 256)|
|WNITER |(1 2 4 8)|
|TM |(4 8 16 32)|
|TN |(4 8 16 32)|

Threads: 1024 Threads per Block, max 1536 threads per SM ⇒ Upper limit 1 block.
1024 Threads per Block, 32 threads per warp, 1024 threads / 32 = 32 warps per block
37 regs per thread
Registers: Register allocation granularity is 256 regs on a warp level, hence rounding up to 1280 regs per warp
1280 regs per warp * 32 warps per block = 40960 regs per block.
Max 65536 regs per SM ⇒ upper limit 1 block.

Each block can use max 48 KB of SMEM, but 65536*4B = 262 KB of register space.


# Experiment Result 
1. Effect of weight quantization on inference quality vs speed
2. Effect of matmul operation (gpu calculation speed improvment and analysis)
3. Effect of multihead attention (speed improvement and anaylsis) 
4. CUDA implementation vs python implementation vs pure C++ implementation 

# Area of improvments:
1. Further study on more efficient implementation of matmul kernel  

Double buffering, for better interleaving of computation and memory loading. For now, see CUTLASS Pipelining. In CUTLASS, double buffering is done on two levels: GMEM ⇒ SMEM, and SMEM ⇒ Registerfile.
Getting rid of SMEM bank conflicts. This can be done by optimizing the data layout in SMEM.
Better understanding the GEMM kernels that are implemented in Triton, by looking at the generated PTX.
Among all the kernels that we have tried, the Nvidia’s library cuBLAS is still the fastest in terms of speed - cuBLAS contains not one single implementation of SGEMM, but hundreds of them. This is also a trade-off between size and speed - the cuBLAS library is 500MB of compiled code.

2. More efficient implementation of multihead attention 

3. Less memory and computing requirement - better quantization weight schemes 
Weight quantization has become a popular approach for such optimizations not only for machine learning frameworks like TensorFlow and PyTorch but also for hardware toolchains like NVIDIA® TensorRT and Xilinx® DNNDK

Currently optional: float16 quantization 
Now we have the option to run gpu calculation with float16 quantization. 
Although it might cause minimal loss in accuracy, compared to float32, it reduces model size by up to half.
It also supports some delegates (e.g. the GPU delegate) which can operate directly on float16 data, resulting in faster execution than float32 computations.
It does not reduce latency as much as a quantization to fixed point math.
By default, a float16 quantized model will "dequantize" the weights values to float32 when run on the CPU. (Note that the GPU delegate will not perform this dequantization, since it can operate on float16 data.)

other potentially better quantization schemes:
1. int8 quantization
get further latency improvements, reductions in peak memory usage, and compatibility with integer only hardware devices or accelerators by making sure all model math is integer quantized.
you need to calibrate or estimate the range, i.e, (min, max) of all floating-point tensors in the model. Unlike constant tensors such as weights and biases, variable tensors such as model input, activations (outputs of intermediate layers) and model output cannot be calibrated unless we run a few inference cycles.
As a result, the converter requires a representative dataset to calibrate them. 
It also comes with a more extreme trade-off between speed and accuracy. 

2. 16x8 quantization
The main advantage of this quantization is that it can improve accuracy significantly, but only slightly increase model size.
activations are quantized based on their range to 16-bits, weights are quantized in 8-bit integer and bias is quantized into 64-bit integer. 
Currently inference is noticeably slower than 8-bit full integer due to the lack of optimized kernel implementation.
Currently it is incompatible with the existing hardware accelerated TFLite delegates. 

4. Non-square matrix support for matmul kernels.
You could easily template the kernel with the tile dimensions as template arguments and instantiate several versions, depending on matrix dimensions. 
For a given architecture, there is probably an optimal tile dimension which balances occupancy and instruction level parallelism. 
The "clever" way to solve this is probably to decompose the matrix multiplication into two operations - the first doing the bulk of the work at the optimal tile size, and the second at a different size for the remaining columns. If the result is going straight back to host memory after the product is completed, the second operation might best be done on the host using an optimised BLAS, overlapped with the GPU kernel. 
