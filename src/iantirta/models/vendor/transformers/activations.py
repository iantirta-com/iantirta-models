# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.
#
# Partial code of transformers, improved by iantirta.com
#
# Copyright 2020 The HuggingFace Team. All rights reserved.
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

from __future__ import annotations

import math
from collections import OrderedDict

import torch
from torch import Tensor, nn

from .integrations.hub_kernels import use_kernel_forward_from_hub


@use_kernel_forward_from_hub("GeLU")
class GELUActivation(nn.Module):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created. For
    information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) This is now written in C in nn.functional
    Also see the Gaussian Error Linear Units paper: https://huggingface.co/papers/1606.08415
    """

    def __init__(self, use_gelu_python: bool = False):
        super().__init__()
        if use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = nn.functional.gelu

    def _gelu_python(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

    def forward(self, input: Tensor) -> Tensor:
        return self.act(input)


class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)


ACT2CLS = {
    "gelu": GELUActivation,
    # "gelu_10": (ClippedGELUActivation, {"min": -10, "max": 10}),
    # "gelu_fast": FastGELUActivation,
    # "gelu_new": NewGELUActivation,
    "gelu_python": (GELUActivation, {"use_gelu_python": True}),
    # "gelu_pytorch_tanh": GELUTanh,
    # "gelu_python_tanh": (GELUTanh, {"use_gelu_tanh_python": True}),
    # "gelu_accurate": AccurateGELUActivation,
    "hardswish": nn.Hardswish,
    # "laplace": LaplaceActivation,
    "leaky_relu": nn.LeakyReLU,
    # "linear": LinearActivation,
    # "mish": MishActivation,
    # "quick_gelu": QuickGELUActivation,
    "relu": nn.ReLU,
    # "relu2": ReLUSquaredActivation,
    "relu6": nn.ReLU6,
    "sigmoid": nn.Sigmoid,
    # "silu": SiLUActivation,
    # "sqrtsoftplus": SqrtSoftplusActivation,
    "swish": nn.SiLU,
    "tanh": nn.Tanh,
    "prelu": nn.PReLU,
    # "xielu": XIELUActivation,
}
ACT2FN = ClassInstantier(ACT2CLS)
