#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <ATen/ATen.h>

#include "src/domdec.cuh"
#include "src/utils.hpp"

template <typename Dtype>
void BalanceCUDA(torch::Tensor nu_basic, torch::Tensor mass_delta, Dtype thresh_step) {

    // Check input size
    int B = nu_basic.size(0);
    int C = nu_basic.size(1);
    int N = nu_basic.size(2);
    if (B*C*N != nu_basic.numel())
    throw std::invalid_argument(
        Formatter() << "Shape mismatch: first two dimensions of nu_basic "
                    << "must be the only non-trivial ones"
    ); 

    // Call cuda kernel
    BalanceKernel(
        B, C, N, nu_basic.data_ptr<Dtype>(), mass_delta.data_ptr<Dtype>(), thresh_step
    );
}

// Instantiate
template void BalanceCUDA<float>(
  torch::Tensor nu_basic, torch::Tensor mass_delta, float thresh_step
);
template void BalanceCUDA<double>(
  torch::Tensor nu_basic, torch::Tensor mass_delta, double thresh_step
);