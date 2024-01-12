#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <ATen/ATen.h>

#include "src/logsumexp.cuh"
#include "src/utils.hpp"

// Template struct to determine the appropriate tensor type based on the input 
template <typename Dtype>
struct TensorTypeSelector {
    static const torch::ScalarType type = torch::kFloat32;
};

// Specialize template for doubles
template <>
struct TensorTypeSelector<double> {
    static const torch::ScalarType type = torch::kFloat64;
};

template <typename Dtype>
torch::Tensor LogSumExpCUDA(torch::Tensor alpha, int M, Dtype dx) {
    // Given alpha ~ (B, N), compute beta ~ (B, M) with entry (b, i) given by
    // log(sum_j exp(alpha_bj - c_ij))
    // where c_ij = (i*dx - j*dx)**2 is computed online

    // Check input size
    int B = alpha.size(0);
    int N = alpha.size(1);
    if (B*N != alpha.numel())
    throw std::invalid_argument(
        Formatter() << "Shape mismatch: first two dimensions of alpha "
                    << "must be the only non-trivial ones"
    );

    // Init tensor of size (B, M)
    auto options = torch::TensorOptions()
        .dtype(TensorTypeSelector<Dtype>::type)
        .device(alpha.device());
    torch::Tensor beta = torch::empty({B, M}, options);    

    // Call cuda kernel
    LogSumExpCUDAKernel(
        B, M, N, alpha.data_ptr<Dtype>(), beta.data_ptr<Dtype>(), dx
    );
    return beta;
}

// Instantiate
template torch::Tensor LogSumExpCUDA<float>(
  torch::Tensor alpha, int M, float dx
);
template torch::Tensor LogSumExpCUDA<double>(
  torch::Tensor alpha, int M, double dx
);

////////////////////////////
// logsumexp arbitrary grid
////////////////////////////

template <typename Dtype>
torch::Tensor LogSumExpCUDA_xyeps(
    torch::Tensor alpha, int M, Dtype dx, Dtype dy, Dtype eps
) {
    // Given alpha ~ (B, N), compute beta ~ (B, M) with entry (b, i) given by
    // log(sum_j exp(alpha_bj - c_ij))
    // where c_ij = (i*dx - j*dx)**2 is computed online

    // Check input size
    int B = alpha.size(0);
    int N = alpha.size(1);
    if (B*N != alpha.numel())
    throw std::invalid_argument(
        Formatter() << "Shape mismatch: first two dimensions of alpha "
                    << "must be the only non-trivial ones"
    );

    // Init tensor of size (B, M)
    auto options = torch::TensorOptions()
        .dtype(TensorTypeSelector<Dtype>::type)
        .device(alpha.device());
    torch::Tensor beta = torch::empty({B, M}, options);    

    // Call cuda kernel
    LogSumExpCUDAKernel_xyeps(
        B, M, N, alpha.data_ptr<Dtype>(), beta.data_ptr<Dtype>(), 
        dx, dy, eps
    );
    return beta;
}

// Instantiate
template torch::Tensor LogSumExpCUDA_xyeps<float>(
  torch::Tensor alpha, int M, 
    float dx, float dy, float eps
);
template torch::Tensor LogSumExpCUDA_xyeps<double>(
  torch::Tensor alpha, int M, 
  double dx, double dy, double eps
);