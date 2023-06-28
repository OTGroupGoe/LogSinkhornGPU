#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <ATen/ATen.h>

#include "src/logsumexp.cuh"
#include "src/utils.hpp"

torch::Tensor LogSumExpCUDA(torch::Tensor alpha, int M, float dx) {
  // Given alpha ~ (B, N), compute beta ~ (B, M) with entry (b, i) given by
  // log(sum_j exp(alpha_bj - c_ij))
  // where c_ij = (i*dx - j*dx)**2 is computed online

  // Check input size
  int B = alpha.size(0);
  int N = alpha.size(1);
  if (B*N != alpha.numel())
    throw std::invalid_argument(Formatter()
                                << "Shape mismatch: first two dimensions of alpha "
                                << "must be the only non-trivial ones");
  
  // Init tensor of size (B, M)
  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(alpha.device());
  torch::Tensor beta = torch::empty({B, M}, options);    

  // Call cuda kernel
  LogSumExpCUDAKernel(B, M, N, alpha.data_ptr<float>(), beta.data_ptr<float>(), dx);

  return beta;
}

torch::Tensor InnerNewtonCUDA(
    int n_iter, float tol, torch::Tensor t, float eps, float lam,
    torch::Tensor lognu, torch::Tensor lognu_nJ, torch::Tensor logKTu
) {
    // check input sizes
    int N = t.numel();
    if (N != lognu.numel())
        throw std::invalid_argument(
            Formatter() << "Shape mismatch: "
                        << "lognu must have same dimensions as t"
        );
    if (N != lognu_nJ.numel())
        throw std::invalid_argument(
            Formatter() << "Shape mismatch: "
                        << "lognu_nJ must have same dimensions as t"
        );
    if (N != logKTu.numel())
        throw std::invalid_argument(
            Formatter() << "Shape mismatch: "
                        << "logKTu must have same dimensions as t"
        );

    // call cuda kernel
    InnerNewtonCUDAKernel(
        n_iter, tol, N, t.data_ptr<float>(), eps, lam,
        lognu.data_ptr<float>(), lognu_nJ.data_ptr<float>(), logKTu.data_ptr<float>()    
    );
    return t;
}