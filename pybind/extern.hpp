#include <ATen/cuda/CUDAContext.h>

template <typename Dtype>
at::Tensor LogSumExpCUDA(at::Tensor alpha, int M, Dtype dx);

template <typename Dtype>
void BalanceCUDA(at::Tensor nu_basic, at::Tensor mass_delta, Dtype thresh_step);

template <typename Dtype>
at::Tensor BasicToCompositeCUDA_2D(
  at::Tensor nu_basic, int w, int h,
  at::Tensor left_in_composite, at::Tensor left_in_basic,
  at::Tensor width_basic, 
  at::Tensor bottom_in_composite, at::Tensor bottom_in_basic,
  at::Tensor height_basic
)