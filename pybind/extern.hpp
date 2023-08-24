#include <ATen/cuda/CUDAContext.h>

template <typename Dtype>
at::Tensor LogSumExpCUDA(at::Tensor alpha, int M, Dtype dx);

template <typename Dtype>
void BalanceCUDA(at::Tensor nu_basic, at::Tensor mass_delta, Dtype thresh_step);