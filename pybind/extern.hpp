#include <ATen/cuda/CUDAContext.h>

template <typename Dtype>
at::Tensor LogSumExpCUDA(at::Tensor alpha, int M, Dtype dx);