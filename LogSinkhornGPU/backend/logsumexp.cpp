#include <torch/extension.h>

// CUDA forward declarations

torch::Tensor LogSumExpCUDAKernel(
    torch::Tensor alpha, 
    int M, 
    torch::Tensor dx, 
    torch::Tensor dy);

// C++ interface

torch::Tensor LogSumExpCUDA(
    torch::Tensor alpha, 
    int M, 
    torch::Tensor dx, 
    torch::Tensor dy) {
  AT_ASSERTM(alpha.device().is_cuda(), "alpha must be a CUDA tensor");
  AT_ASSERTM(alpha.is_contiguous(), "alpha must be contiguous");
  AT_ASSERTM(dx.device().is_cpu(), "dx must be a CPU tensor");

  return LogSumExpCUDAKernel(alpha, M, dx, dy);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("LogSumExpCUDA", &LogSumExpCUDA, "LogSumExpCUDA");
}
