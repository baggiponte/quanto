#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "gemm/gemm_cuda.h"
#include "gemv/gemv_cuda.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("gemm_forward_cuda_new", &gemm_forward_cuda_new, "New quantized GEMM kernel.");
    m.def("gemv_forward_cuda_new", &gemv_forward_cuda_new, "New quantized GEMV kernel.");
}
