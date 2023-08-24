#include <string>

#include <torch/extension.h>

#include "pybind/extern.hpp"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("LogSumExpCUDA_32", &LogSumExpCUDA<float>, "LogSumExpCUDA (float)"); 
  m.def("LogSumExpCUDA_64", &LogSumExpCUDA<double>, "LogSumExpCUDA (double)"); 
  m.def("BalanceCUDA_32", &BalanceCUDA<float>, "BalanceCUDA (float)"); 
  m.def("BalanceCUDA_64", &BalanceCUDA<double>, "BalanceCUDA (double)"); 
  m.def("BasicToCompositeCUDA_2D_32", &BasicToCompositeCUDA_2D<float>, 
        "BasicToCompositeCUDA_2D (float)"); 
  m.def("BasicToCompositeCUDA_2D_64", &BasicToCompositeCUDA_2D<double>, 
        "BalanceBasicToCompositeCUDA_2D (double)"); 
  //m.def("LogSumExpGPU_accesor", &LogSumExpGPU_accesor);
}
