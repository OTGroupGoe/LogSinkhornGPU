#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

template <typename Dtype>
void BalanceKernel(
    int B, int C, int N, Dtype *nu_basic, Dtype *mass_delta, Dtype thresh_step
);

template <typename Dtype>
void BasicToCompositeKernel_2D(
    int B, int C,
    torch::PackedTensorAccessor32<Dtype, 3> nu_composite,
    torch::PackedTensorAccessor32<Dtype, 4> nu_basic,
    torch::PackedTensorAccessor32<int, 2> left_in_composite,
    torch::PackedTensorAccessor32<int, 2> left_in_basic,
    torch::PackedTensorAccessor32<int, 2> width_basic,
    torch::PackedTensorAccessor32<int, 2> bottom_in_composite,
    torch::PackedTensorAccessor32<int, 2> bottom_in_basic,
    torch::PackedTensorAccessor32<int, 2> height_basic
);

template <typename Dtype>
void AddWithOffsetsKernel_2D(
    int B, int C,
    torch::PackedTensorAccessor32<Dtype, 3> nu_composite,
    torch::PackedTensorAccessor32<Dtype, 3> nu_basic,
    torch::PackedTensorAccessor32<Dtype, 3> weights,
    torch::PackedTensorAccessor32<int, 2> sum_indices,
    torch::PackedTensorAccessor32<int, 2> left_in_composite,
    torch::PackedTensorAccessor32<int, 2> left_in_basic,
    torch::PackedTensorAccessor32<int, 2> width_basic,
    torch::PackedTensorAccessor32<int, 2> bottom_in_composite,
    torch::PackedTensorAccessor32<int, 2> bottom_in_basic,
    torch::PackedTensorAccessor32<int, 2> height_basic
);