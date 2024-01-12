#include <ATen/cuda/CUDAContext.h>

template <typename Dtype>
at::Tensor LogSumExpCUDA(at::Tensor alpha, int M, Dtype dx);

template <typename Dtype>
at::Tensor LogSumExpCUDA_xyeps(at::Tensor alpha, int M, 
    Dtype dx, Dtype dy, Dtype eps
);