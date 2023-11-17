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

# Changes (until Nov.1)
1. Added possible optimizations for neural net operations in readme   
2. Wrote code for kernels calculating rotationary embedding and RMSNorm.
  
# Changes (until Nov.7)
1. Added siLU kernel 
2. Copied function for calculating time from llama2.cpp. 
3. Added naive implementation of multihead attention kernel which calculates the normalized dot product of K transpose and Q, and then store the result of another dot product with V. Added function declarations in run.cu and experiments with other more efficient implementation of multihead attention will be explored if time permitting.
4. Modified run instruction in readme. 
5. Added code in main function to read tokenizer.bin file which can be generated through python script.
6. Wrote naive implementation of elementwise add kernel function and accum_gpu function that calls that kernel function. Other more efficient implementations of elementwise add will be explored if time permitting.

# Changes (until Nov.10)
1. Added more efficient implementations of matmul operations, including global memory coalescing, shared memory cache-blocking, 1d/2d blocktiling, etc. 
2. Moved kernel functions of matmul operations to another folder matmul_kernels for cleaness, added kernels.cu file to include those kernel functions. 
3. Added function in run.cu to get GPU info for calculations, including max threads per block, total global memory/shared memory, multiprocessor count, etc.
4. Modified readme to include more efficient implementations of matmul. Updated progress tracker.
5. Added softmax_gpu in run.cu.

# Changes (until Nov.11)
1. Changed run.h to run.cuh, modified paths to files in kernels.cu
2. Added run_kernel function in run.cu to select the desired matmul kernel function to run based on kernel number.
3. Added optional weight quantization support - added weight quantization kernel for converting float back to half after gpu calculation is done in run.cu, still working on minor adjustments in gpu kernels for converting float to half on gpu.
4. Added transformer function in main function in run.cu that calls all the kernel functions and generates output based on weights and input. 
5. Added python util folder and plot_benchmark_results.py to generate experiment results.
6. Updated run instruction to include a pretrainded transformer model to download for inference. 

# Changes (until Nov.12)
1. Added support for half precision calculation in matmul kernels - completed code for 1d blocktiling, 2d blocktiling, and vectorized kernel.  
2. Added another matmul kernel using cub:reduce lib for comparison.
3. Updated TODO tracker to show that two parts have been completed - more efficient implementations of matmul kernels and write transformer architecture. Some efficient implementations of matmul only works for square matrices at this moment, and will be adapted to specific matrix shape in the transformer if time permitting. 

# Changes (until Nov.13)
1. Added support for half precision calculation in add kernel.
2. Added utility functions argmax and sample in . Updated TODO tracker correspondingly.

# Changes (until Nov.14)
1. Added result section with placeholders for experiment data and figures.
2. Moved changes log from readme.md to log.md.

# Changes (until Nov.15)
1. Added sgemm_run.cu to test the matmul kernels with squared matrices.
2. Fixed a bug - added cublas destory in run_sgemm_matmul_kernel function in run.cu.
3. Added gen_results.sh to call each kernel function and then plot the results at the end.

# Changes (until Nov.16)
1. Added table to summarize the Block Number /Thread Number/ Element per thread used by each kernel
2. Added introduction to each kernel method in matmul_kernels.txt. 
3. Added areas of improvement and possible methods in report.txt.

# Changes (until Nov.17)
1. Added CMakeList.txt with two executable files to build: sgemm and inference. The first one tests the matmul kernels and the second one runs the inference with kernel number and weight quantization number specified and calculates time.
2. Created and moved source code to src folder except run.cu and sgemm_run.cu.

# Changes (until Nov.18)
1. Moved runner functions in run.cu to /src/run.cu, and renamed original run.cu to inference.cu.
2. Modified input type from float* to void* for all kernel functions to accommodate half/full precision calculation.
3. Fixed errors when calling matmul kernel functions with template specification.
4. Added check_params.cpp to check if kernel functions have received valid input dimensions. 
5. Changed log.txt to log.md, added result tables and parameter informations to it.

# Changes (until Nov.19)
1. Changed softmax function to __device__ type function that only gets called by transformer function and moved it to transformer_kernels folder. Wrote another cpu version of softmax in run.cu. 
2. Fixed mismatches between function declaration and definitions in run.cu and run.cuh 
3. Moved runner functions from inference.cu to run.cu to match the function declarations in run.cuh.
4. Added sgemm_run_test.cu to test simple model loading and matmul kernel calling.
5. Added training procedure in readme for customized model.


