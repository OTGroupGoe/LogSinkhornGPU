#include <torch/extension.h>

// CUDA forward declarations

torch::Tensor LogSumExpCUDAKernel(
    torch::Tensor alpha, 
    int M, 
    torch::Tensor dx, 
    torch::Tensor dy);

void BalanceKernel(
    torch::Tensor nu_basic, 
    torch::Tensor mass_delta, 
    torch::Tensor thresh_step);

torch::Tensor AddWithOffsetsKernel_2D(
    torch::Tensor nu_basic, int w, int h,
    torch::Tensor weights, torch::Tensor sum_indices,
    torch::Tensor left_in_composite, torch::Tensor left_in_basic,
    torch::Tensor width_basic,
    torch::Tensor bottom_in_composite, torch::Tensor bottom_in_basic,
    torch::Tensor height_basic);

void InnerNewtonCUDAKernel(
    int n_iter, torch::Tensor tol, torch::Tensor eps, torch::Tensor lam,
    torch::Tensor t, torch::Tensor lognu, 
    torch::Tensor lognu_nJ, torch::Tensor logKTu);

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

void BalanceCUDA(torch::Tensor nu_basic, torch::Tensor mass_delta, 
                torch::Tensor thresh_step)
{

  // Check nu_basic is 3-dimensional: 0 batch, 1 basic-in-composite, 2 physical
  AT_ASSERTM(nu_basic.dim() == 3, "nu_basic must be 3-dimensional");

  // Call cuda kernel
  BalanceKernel(nu_basic, mass_delta, thresh_step);
}

/////////////////////
// add with offsets
/////////////////////

torch::Tensor AddWithOffsetsCUDA_2D(
    torch::Tensor nu_basic, int w, int h,
    torch::Tensor weights, torch::Tensor sum_indices,
    torch::Tensor left_in_composite, torch::Tensor left_in_basic,
    torch::Tensor width_basic,
    torch::Tensor bottom_in_composite, torch::Tensor bottom_in_basic,
    torch::Tensor height_basic)
{

  // TODO: should we check anything here?

  // Call CUDA function
  return AddWithOffsetsKernel_2D(nu_basic, w, h,
    weights, sum_indices,
    left_in_composite, left_in_basic,
    width_basic,
    bottom_in_composite, bottom_in_basic,
    height_basic
  );
}

void InnerNewtonCUDA(
    int n_iter, torch::Tensor tol, torch::Tensor eps, torch::Tensor lam,
    torch::Tensor t, torch::Tensor lognu, torch::Tensor lognu_nJ, torch::Tensor logKTu)

  {
  // TODO: should we check anything here?

  // Call CUDA Kernel
  InnerNewtonCUDAKernel(n_iter, tol, eps, lam, t, lognu, lognu_nJ, logKTu);
  }


// PYBIND call
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("LogSumExpCUDA", &LogSumExpCUDA, "LogSumExpCUDA");
  m.def("BalanceCUDA", &BalanceCUDA, "BalanceCUDA");
  m.def("AddWithOffsetsCUDA_2D", &AddWithOffsetsCUDA_2D, "AddWithOffsetsCUDA_2D");
  m.def("InnerNewtonCUDA", &InnerNewtonCUDA, "InnerNewtonCUDA");
}
