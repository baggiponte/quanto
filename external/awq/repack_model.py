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
import argparse
import time
import torch

from transformers import AutoModelForCausalLM
from pack_intweight import pack_intweight
from pack import pack_awq
from quanto import QLinear, freeze, qint4, quantize


def main():
    parser = argparse.ArgumentParser(description="Evaluate model quantization and repacking time")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-350m",
        help="The name of the model to quantize and repack.",
    )
    parser.add_argument("--device", type=str, default=None, help="The device to use for generation.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(
        device
    )
    start = time.time()
    quantize(model, weights=qint4)
    end = time.time()
    print(f"Model quantized to qint4 in {end - start:.2f} s")
    start = time.time()
    freeze(model)
    end = time.time()
    print(f"Model frozen in {end - start:.2f} s")
    start = time.time()
    for name, m in model.named_modules():
        if isinstance(m, QLinear):
            print(f"Unpacking and repacking {name}")
            unpacked = m.weight._data.unpack()
            try:
                # unpacked should be reshaped to have last dim a multiple of 32
                packed = pack_awq(unpacked, interleave=4, kstride=64)
            except:
                pass
    end = time.time()
    print(f"Model repacked in {end - start:.2f} s")

if __name__ == "__main__":
    main()
