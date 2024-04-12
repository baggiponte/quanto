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
import torch


def pack_awq(unpacked, interleave=4, kstride=64):
    """Pack tensor as expected by AWQ fast mixed mm kernel

    The packing aims at two things:

    - permute rows as expected on Turing and Ampere architecture for faster dequantization
    - interleave groups of 'interleave' rows for efficient parallel processing
    """
    # unpacked: [N, K]
    N = unpacked.shape[0]
    K = unpacked.shape[1]
    I = interleave
    S = kstride

    # 1. For faster dequantization, the tensor rows must be permuted as explained in:
    # https://github.com/NVIDIA/TensorRT-LLM/blob/035b99e0d09d4f2dfdb949810cf7245112aa4165/cpp/tensorrt_llm/kernels/cutlass_kernels/cutlass_preprocessors.cpp#L161
    packed = unpacked.reshape(N, K // 32, 32)
    # np.arange(32).reshape(4, 4, 2).transpose(1, 0, 2) => [0, 1, 8, 9, 16, 17, 24, 25, ...]
    packed = packed.reshape(N, K // 32, 4, 4, 2).permute(0, 1, 3, 2, 4)
    packed = packed.reshape(N, K // 32, 32)

    # reorder each 8 weights for fast dequantization
    # [0, 1, 2, 3, 4, 5, 6, 7] => [0, 2, 4, 6, 1, 3, 5, 7]
    packed = packed.reshape(N, K // 32, 4, 8)
    packed = packed.reshape(N, K // 32, 4, 4, 2).permute(0, 1, 2, 4, 3)
    packed = packed.reshape(N, K)

    # 2. For efficient parallelization, the rows are grouped and interleaved by blocks of kstride into a single row, as explained in:
    # https://github.com/NVIDIA/TensorRT-LLM/blob/d37b507f41a87457fe9f10f7459d08f5db235745/cpp/tensorrt_llm/kernels/weightOnlyBatchedGemv/kernel.h#L69
    # interleaving (N, K) -> (N // I, I, K // S, S)
    packed = packed.reshape(N // I, I, K // S, S)
    # transpose (N // I, I, K // S, S) -> (N // I, K // S, I, S)
    packed = packed.permute(0, 2, 1, 3)
    # reshape (N // I, K // S, I, S) -> (N // I, K // S, S, I)
    packed = packed.reshape(N // I, K // S, S, I)
    # Packing (N // I, K // S, S, I) -> (N // I, K // S, S)
    packed = packed.to(torch.int32)
    def lshift(t: torch.Tensor, bits: int):
        if t.device.type == "mps":
            # lshift is not supported on MPS device
            return t * (2**bits)
        return t << bits
    packed = (
        packed[..., 0]
        | lshift(packed[..., 1], 4)
        | lshift(packed[..., 2], 8)
        | lshift(packed[..., 3], 12)
    )
    # Reshape to (N // I, K // S, S) -> (N // I, K)
    packed = packed.reshape(N // I, K)
    packed = packed.to(torch.int16).to(unpacked.device).contiguous()
    return packed
