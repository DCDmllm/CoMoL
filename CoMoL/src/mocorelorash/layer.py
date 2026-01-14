"""
MoCoreLORA Layer
"""
import math
from abc import ABC
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# from ..lora import LoraLayer

from peft.tuners.tuners_utils import BaseTunerLayer

class MoCoreLoraLayer(BaseTunerLayer, ABC):
    """
    MoCoreLoRA Layer
    """

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.lora_rank = {}
        self.lora_alpha = {}
        self.scaling = {}

        self.lora_dropout = nn.ModuleDict({})
        # self.lora_A = nn.ModuleDict({})
        self.lora_Cores = nn.ParameterDict({})
        # self.lora_B = nn.ModuleDict({})
        
        self.lora_router = nn.ModuleDict({})

        self.kwargs = kwargs

        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

        self.in_features = in_features
        self.out_features = out_features

    def update_layer(
        self, adapter_name: str, lora_A: nn.Linear, lora_B: nn.Linear, lora_rank: int, lora_alpha: int, lora_dropout: float, init_lora_weights: bool,
        num_experts: int, core_router: bool = False
    ) -> None:
        """
        Update the layer
        """
        if lora_rank <= 0:
            raise ValueError(f"The rank `r` should be a positive integer value but the value passed is {lora_rank}.")

        self.lora_rank[adapter_name] = lora_rank
        self.lora_alpha[adapter_name] = lora_alpha

        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer
        # self.lora_A[adapter_name] = nn.Linear(self.in_features, lora_rank, bias=False)
        self.lora_A = lora_A
        self.lora_B = lora_B
        self.lora_Cores[adapter_name] = nn.ParameterList([torch.randn(lora_rank, lora_rank) for _ in range(num_experts)])  # 初始化比较重要
        # self.lora_B[adapter_name] = nn.Linear(lora_rank, self.out_features, bias=False)
        self.scaling[adapter_name] = lora_alpha / lora_rank

        if core_router:
            self.lora_router[adapter_name] = nn.Linear(lora_rank, num_experts, bias=False)
        else:
            self.lora_router[adapter_name] = nn.Linear(self.in_features, num_experts, bias=False)

        self.reset_parameters(adapter_name, init_lora_weights)
        self.set_adapter(self.active_adapters)

    def reset_parameters(self, adapter_name: str, init_lora_weights: bool) -> None:
        """
        Reset the parameters
        """
        if init_lora_weights is False:
            return
        elif adapter_name in self.lora_Cores.keys():
            # nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
            for i in range(len(self.lora_Cores[adapter_name])):
                nn.init.kaiming_uniform_(self.lora_Cores[adapter_name][i], a=math.sqrt(5))
            # nn.init.zeros_(self.lora_B[adapter_name].weight)


class LinearMoCoreLoraLayer(nn.Module, MoCoreLoraLayer):
    """
    MoCoreLoRA Implementation in a Linear Layer
    """

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        lora_A: nn.Linear,
        lora_B: nn.Linear,
        lora_rank: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: bool = True,
        num_experts: int = 8,
        core_router: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        MoCoreLoraLayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.core_router = core_router
        self.update_layer(adapter_name, lora_A, lora_B, lora_rank, lora_alpha, lora_dropout, init_lora_weights, num_experts, core_router)

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
            if active_adapter not in self.lora_Cores.keys():
                continue

            # lora_A = self.lora_A[active_adapter]
            lora_A = self.lora_A
            lora_Cores = self.lora_Cores[active_adapter]
            # lora_B = self.lora_B[active_adapter]
            lora_B = self.lora_B
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            lora_router = self.lora_router[active_adapter]

            x = x.to(lora_A.weight.dtype)

            lora_A_x = lora_A(dropout(x))
            if self.core_router:
                router_logits = F.softmax(lora_router(lora_A_x), dim=-1)
            else:
                router_logits = F.softmax(lora_router(x), dim=-1)
            # fuse cores
            lora_Core = torch.sum(torch.stack([core * router_logits[:, :, i].unsqueeze(-1).unsqueeze(-1) for i, core in enumerate(lora_Cores)]), dim=0)
            # print('lora_Core.shape', lora_Core.shape)
            result += lora_B((lora_A_x.unsqueeze(-2) @ lora_Core).squeeze(-2)) * scaling
            
            # router_logits = F.softmax(lora_router(x), dim=-1)
            # # fuse cores
            # lora_Core = torch.sum(torch.stack([core * router_logits[:, :, i].unsqueeze(-1).unsqueeze(-1) for i, core in enumerate(lora_Cores)]), dim=0)
            # # print('lora_Core.shape', lora_Core.shape)
            # result += lora_B((lora_A(dropout(x)).unsqueeze(-2) @ lora_Core).squeeze(-2)) * scaling

        result = result.to(previous_dtype)
        return result
