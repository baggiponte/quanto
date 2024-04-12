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
import numpy as np
import torch


def unpack_awq(packed, interleave, kstride):
    N = packed.shape[0] * interleave
    K = packed.shape[1]
    I = interleave
    S = kstride
    # Reshape (N // I, K) -> (N // I, K // S, S, 1)
    unpacked = packed.reshape(N // I, K // S, S, 1)
    # Convert to uint16 (through numpy because not supported by pytorch)
    unpacked = unpacked.cpu().numpy().astype(np.uint16)
    # Unpack (N // I, K, S) -> (N // I, K // S, S, I)
    unpacked = torch.cat([
        torch.tensor((unpacked & 0xF).astype(np.uint8)).to(packed.device),
        torch.tensor(((unpacked & 0xF0) >> 4).astype(np.uint8)).to(packed.device),
        torch.tensor(((unpacked & 0xF00) >> 8).astype(np.uint8)).to(packed.device),
        torch.tensor(((unpacked & 0XF000) >> 12).astype(np.uint8)).to(packed.device)
    ], axis=-1)
    # reshape (N // I, K // S, S, I) -> (N // I, K // S, I, S)
    unpacked = unpacked.reshape(N // I, K // S, I, S)
    # transpose (N // I, K // S, I, S) -> (N // I, I, K // S, S)
    unpacked = unpacked.permute(0, 2, 1, 3)
    # deinterleaving (N // I, I, K // S, S) -> (N, K)
    unpacked = unpacked.reshape(N, K)

    # Final unexplained steps to reorder
    unpacked = unpacked.reshape(N, K // 32, 4, 2, 4).permute(0, 1, 2, 4, 3)
    unpacked = unpacked.reshape(N, K // 32, 4, 8)

    unpacked = unpacked.reshape(N, K // 32, 32)
    unpacked = unpacked.reshape(N, K // 32, 4, 4, 2).permute(0, 1, 3, 2, 4)
    unpacked = unpacked.reshape(N, K // 32, 32)
    unpacked = unpacked.reshape(N, K)

    return unpacked
