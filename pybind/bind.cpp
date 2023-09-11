#include <string>

#include <torch/extension.h>

#include "pybind/extern.hpp"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("LogSumExpCUDA_32", &LogSumExpCUDA<float>, "LogSumExpCUDA (float)"); 
  m.def("LogSumExpCUDA_64", &LogSumExpCUDA<double>, "LogSumExpCUDA (double)"); 
  //m.def("LogSumExpGPU_accesor", &LogSumExpGPU_accesor);
  m.def("InnerNewtonCUDA", &InnerNewtonCUDA);
}
