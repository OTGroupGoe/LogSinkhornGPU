#include <ATen/cuda/CUDAContext.h>

template <typename Dtype>
at::Tensor LogSumExpCUDA(at::Tensor alpha, int M, Dtype dx);
at::Tensor InnerNewtonCUDA(
    int n_iter, float tol, at::Tensor t, float eps, float lam,
    at::Tensor lognu, at::Tensor lognu_nJ, at::Tensor logKTu
);

template <typename Dtype>
void BalanceCUDA(at::Tensor nu_basic, at::Tensor mass_delta, Dtype thresh_step);

template <typename Dtype>
at::Tensor BasicToCompositeCUDA_2D(
  at::Tensor nu_basic, int w, int h,
  at::Tensor left_in_composite, at::Tensor left_in_basic,
  at::Tensor width_basic, 
  at::Tensor bottom_in_composite, at::Tensor bottom_in_basic,
  at::Tensor height_basic
);

template <typename Dtype>
torch::Tensor AddWithOffsetsCUDA_2D(
  torch::Tensor nu_basic, int w, int h,
  torch::Tensor weights, torch::Tensor sum_indices,
  torch::Tensor left_in_composite, torch::Tensor left_in_basic,
  torch::Tensor width_basic, 
  torch::Tensor bottom_in_composite, torch::Tensor bottom_in_basic,
  torch::Tensor height_basic
);