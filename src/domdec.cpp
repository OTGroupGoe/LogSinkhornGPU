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

/////////////////////
// basic to composite
/////////////////////

template <typename Dtype>
torch::Tensor BasicToCompositeCUDA_2D(
  torch::Tensor nu_basic, int w, int h,
  torch::Tensor left_in_composite, torch::Tensor left_in_basic,
  torch::Tensor width_basic, 
  torch::Tensor bottom_in_composite, torch::Tensor bottom_in_basic,
  torch::Tensor height_basic
)
{
  int B = nu_basic.size(0);
  int C = nu_basic.size(1);

  // init composite tensor
  auto options = torch::TensorOptions() 
          .dtype(TensorTypeSelector<Dtype>::type)
          .device(nu_basic.device());
  torch::Tensor nu_composite = torch::empty({B, w, h}, options);

  // Pass accesors
  BasicToCompositeKernel_2D(
      B, C, 
      nu_composite.packed_accessor32<Dtype,3>(),
      nu_basic.packed_accessor32<Dtype, 4>(), 
      left_in_composite.packed_accessor32<Dtype, 2>(), 
      left_in_basic.packed_accessor32<Dtype, 2>(), 
      width_basic.packed_accessor32<Dtype, 2>(), 
      bottom_in_composite.packed_accessor32<Dtype, 2>(), 
      bottom_in_basic.packed_accessor32<Dtype, 2>(), 
      height_basic.packed_accessor32<Dtype, 2>() 
  );
  return nu_composite;
}

// Instantiate

template torch::Tensor BasicToCompositeCUDA_2D<float>(
  torch::Tensor nu_basic, int w, int h,
  torch::Tensor left_in_composite, torch::Tensor left_in_basic,
  torch::Tensor width_basic, 
  torch::Tensor bottom_in_composite, torch::Tensor bottom_in_basic,
  torch::Tensor height_basic
)

template torch::Tensor BasicToCompositeCUDA_2D<double>(
  torch::Tensor nu_basic, int w, int h,
  torch::Tensor left_in_composite, torch::Tensor left_in_basic,
  torch::Tensor width_basic, 
  torch::Tensor bottom_in_composite, torch::Tensor bottom_in_basic,
  torch::Tensor height_basic
)