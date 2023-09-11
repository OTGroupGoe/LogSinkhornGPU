#include <ATen/cuda/CUDAContext.h>

template <typename Dtype>
at::Tensor LogSumExpCUDA(at::Tensor alpha, int M, Dtype dx);
at::Tensor InnerNewtonCUDA(
    int n_iter, float tol, at::Tensor t, float eps, float lam,
    at::Tensor lognu, at::Tensor lognu_nJ, at::Tensor logKTu
);