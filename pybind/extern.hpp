#include <ATen/cuda/CUDAContext.h>

at::Tensor LogSumExpCUDA(at::Tensor alpha, int M, float dx);