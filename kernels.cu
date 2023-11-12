#pragma once

#include "matmul_kernels/1_naive.cu"
#include "matmul_kernels/2_global_mem_coalesce.cu"
#include "matmul_kernels/3_shared_mem_blocking.cu"
#include "matmul_kernels/4_1d_blocktiling.cu"
#include "matmul_kernels/5_2d_blocktiling.cu"
#include "matmul_kernels/6_vectorized.cu"
#include "matmul_kernels/7_warptiling.cu"

#include "transformer_kernels/RoPERotationKernel.cu"
#include "transformer_kernels/multiHeadAttentionKernel_naive.cu"
#include "transformer_kernels/siluElementwiseMulKernel.cu"

#include "add_kernels/elementwiseAddKernel.cu"

#include "weight_quantization_kernels/half_to_full.cu" 