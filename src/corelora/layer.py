"""
CoreLORA Layer
"""
import math
from abc import ABC
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# from ..lora import LoraLayer

from peft.tuners.tuners_utils import BaseTunerLayer

class CoreLoraLayer(BaseTunerLayer, ABC):
    """
    CoreLoRA Layer
    """

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.lora_rank_d = {}
        self.lora_rank_u = {}
        self.lora_alpha = {}
        self.scaling = {}

        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_Core = nn.ParameterDict({})
        self.lora_B = nn.ModuleDict({})
        self.kwargs = kwargs

        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

        self.in_features = in_features
        self.out_features = out_features

    def update_layer(
        self, adapter_name: str, lora_rank_d: int, lora_rank_u: int, lora_alpha: int, lora_dropout: float, init_lora_weights: bool,
    ) -> None:
        """
        Update the layer
        """
        if lora_alpha % ((lora_rank_d + lora_rank_u) // 2) != 0:
            raise ValueError(f"lora_alpha should be divisible by lora rank d+u // 2 ")

        self.lora_rank_d[adapter_name] = lora_rank_d
        self.lora_rank_u[adapter_name] = lora_rank_u
        self.lora_alpha[adapter_name] = lora_alpha

        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer
        self.lora_A[adapter_name] = nn.Linear(self.in_features, lora_rank_d, bias=False)
        self.lora_Core[adapter_name] = nn.Parameter(torch.randn(lora_rank_d, lora_rank_u))  # 初始化比较重要
        self.lora_B[adapter_name] = nn.Linear(lora_rank_u, self.out_features, bias=False)
        self.scaling[adapter_name] = lora_alpha / ((lora_rank_d + lora_rank_u) // 2)

        self.reset_parameters(adapter_name, init_lora_weights)
        self.set_adapter(self.active_adapters)

    def reset_parameters(self, adapter_name: str, init_lora_weights: bool) -> None:
        """
        Reset the parameters
        """
        if init_lora_weights is False:
            return
        elif adapter_name in self.lora_A.keys():
            nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_Core[adapter_name], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[adapter_name].weight)

            # train core, freeze A and B
            # nn.init.zeros_(self.lora_Core[adapter_name])
            # nn.init.kaiming_uniform_(self.lora_B[adapter_name].weight, a=math.sqrt(5))
            # self.lora_A[adapter_name].weight.requires_grad = False
            # self.lora_B[adapter_name].weight.requires_grad = False


class LinearCoreLoraLayer(nn.Module, CoreLoraLayer):
    """
    CoreLoRA Implementation in a Linear Layer
    """

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        lora_rank_d: int = 0,
        lora_rank_u: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        CoreLoraLayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, lora_rank_d, lora_rank_u, lora_alpha, lora_dropout, init_lora_weights)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights inside the base weights
        """
        raise NotImplementedError

    def unmerge(self) -> None:
        """
        Unmerge all merged adapter layers from the base weights
        """
        raise NotImplementedError

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward propagation
        """
        previous_dtype = x.dtype
        result = self.base_layer(x, *args, **kwargs)

        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue

            lora_A = self.lora_A[active_adapter]
            lora_Core = self.lora_Core[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            x = x.to(lora_A.weight.dtype)
            result += lora_B(lora_A(dropout(x)) @ lora_Core) * scaling
            # x = lora_A(dropout(x))
            # x = x @ lora_Core
            # x = lora_B(x) * scaling
            # result += x

        result = result.to(previous_dtype)
        return result
