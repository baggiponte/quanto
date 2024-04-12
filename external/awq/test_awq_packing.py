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
import pytest
import torch

from pack_intweight import pack_intweight
from pack import pack_awq
from unpack import unpack_awq


@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("random", [True, False])
def test_pack_tensor(shape, random, device):
    """This test verifies two things:

    - that we are able to replicate awq packing,
    - that we can unpack awq packed tensors and recover the original tensor.
    """
    bits = 4
    interleave = 4
    kstride = 64
    qmax = 2**bits
    if random:
        t = torch.randint(0, qmax, shape, dtype=torch.uint8).to(device)
    else:
        numel = np.prod(shape)
        t = torch.tensor(range(numel), dtype=torch.int32)
        t = (t % qmax).reshape(shape).to(torch.uint8).to(device)
    packed = pack_intweight(t.to(torch.int32), interleave=interleave, kstride=kstride)
    repacked = pack_awq(t, interleave=interleave, kstride=kstride)
    assert torch.equal(packed, repacked)
    unpacked = unpack_awq(packed, interleave=interleave, kstride=kstride)
    assert torch.equal(unpacked, t)

