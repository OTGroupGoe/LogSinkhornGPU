#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <ATen/ATen.h>

#include "src/logsumexp.cuh"
#include "src/utils.hpp"

torch::Tensor LogSumExpCUDA(torch::Tensor alpha, int M, float dx) {
  // Make beta of size B, M
  int B = alpha.size(0);
  int N = alpha.size(1);
  if (B*N != alpha.numel())
    throw std::invalid_argument(Formatter()
                                << "Shape mismatch: first two dimensions of alpha "
                                << "must be the only non-trivial ones");
  
  // at::IntArrayRef size_beta = {B, M};
  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(alpha.device());
  torch::Tensor beta = torch::empty({B, M}, options);    

  LogSumExpCUDAKernel(B, M, N, alpha.data_ptr<float>(), beta.data_ptr<float>(), dx);

  return beta;
}