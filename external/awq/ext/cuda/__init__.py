# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import torch
from torch.utils.cpp_extension import load


__all__ = []

extra_cflags = ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16"]
extra_cuda_cflags = [
        "-O3",
        "-std=c++17",
        "-DENABLE_BF16",  # TODO
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "--threads=8",
    ]
module_path = os.path.dirname(__file__)
awq_cuda = load(
    name="awq_cuda",
    sources=[
        f"{module_path}/gemm/gemm_cuda.cu",
        f"{module_path}/gemv/gemv_cuda.cu",
        f"{module_path}/pybind_module.cpp",
    ],
    extra_cflags=extra_cflags,
    extra_cuda_cflags=extra_cuda_cflags,
    verbose=True
)


@torch.library.impl("awq::gemm", ["CUDA"])
def gemm_cuda(input: torch.Tensor, other: torch.Tensor, scales: torch.Tensor, zeropoint: torch.Tensor):
    return awq_cuda.gemm_forward_cuda_new(input, other, scales, zeropoint)


@torch.library.impl("awq::gemv", ["CUDA"])
def gemv_cuda(input: torch.Tensor, other: torch.Tensor, scales: torch.Tensor, zeropoint: torch.Tensor, m: int, n: int, k: int, group_size: int):
    return awq_cuda.gemv_forward_cuda_new(input, other, scales, zeropoint, m, n, k, group_size)
