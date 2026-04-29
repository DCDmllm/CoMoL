"""
MoLE Layer
"""
import math
from abc import ABC
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..lora import LoraLayer


class MoLoraLayer(LoraLayer, ABC):
    """
    MoLE Layer
    """

    def __init__(self, base_layer: nn.Module, **kwargs):
        super().__init__(base_layer, **kwargs)
        self.lora_gating = nn.ModuleDict({})
        # self.moe_layer = nn.ModuleDict({})

    def update_layer(
        self, adapter_name: str, lora_rank: int, lora_alpha: int, lora_dropout: float, init_lora_weights: bool,
        num_experts: int, hydra: bool
    ) -> None:
        """
        Update the layer
        """
        if lora_rank <= 0:
            raise ValueError(f"The rank `r` should be a positive integer value but the value passed is {lora_rank}.")

        self.lora_rank[adapter_name] = lora_rank
        self.lora_alpha[adapter_name] = lora_alpha

        if lora_dropout > 0.0:
            if hydra:
                lora_dropout_layer = nn.Dropout(p=lora_dropout)
            else:
                lora_dropout_layer = nn.ModuleList(nn.Dropout(p=lora_dropout) for _ in range(num_experts))
        else:
            if hydra:
                lora_dropout_layer = nn.Identity(p=lora_dropout)
            else:
                lora_dropout_layer = nn.ModuleList(nn.Identity(p=lora_dropout) for _ in range(num_experts))

        self.lora_dropout[adapter_name] = lora_dropout_layer

        if hydra:
            self.lora_A[adapter_name] = nn.Linear(self.in_features, lora_rank, bias=False)
        else:
            self.lora_A[adapter_name] = nn.ModuleList(
                nn.Linear(self.in_features, lora_rank, bias=False) for _ in range(num_experts))

        self.lora_B[adapter_name] = nn.ModuleList(
            nn.Linear(lora_rank, self.out_features, bias=False) for _ in range(num_experts))
        self.scaling[adapter_name] = lora_alpha / lora_rank
        self.lora_gating[adapter_name] = nn.Linear(self.in_features, num_experts, bias=False)

        self.reset_parameters(adapter_name, init_lora_weights, hydra)
        self.set_adapter(self.active_adapters)

    def reset_parameters(self, adapter_name: str, init_lora_weights: bool, hydra:bool) -> None:
        """
        Reset the parameters
        """
        if init_lora_weights is False:
            return
        elif adapter_name in self.lora_A.keys():
            if hydra:
                nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
            else:
                for i in range(len(self.lora_A[adapter_name])):
                    nn.init.kaiming_uniform_(self.lora_A[adapter_name][i].weight, a=math.sqrt(5))
            for i in range(len(self.lora_B[adapter_name])):
                nn.init.zeros_(self.lora_B[adapter_name][i].weight)


class LinearMoLoraLayer(nn.Module, MoLoraLayer):
    """
    MoLE Implementation in a Linear Layer
    """

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        lora_rank: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: bool = True,
        num_experts: int = 4,
        hydra: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        MoLoraLayer.__init__(self, base_layer=base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name, lora_rank, lora_alpha, lora_dropout, init_lora_weights, num_experts, hydra)
        self.hydra = hydra

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights inside the base weights
        """
        pass

    def unmerge(self) -> None:
        """
        Unmerge all merged adapter layers from the base weights
        """
        pass

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
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            lora_gating = self.lora_gating[active_adapter]
            scaling = self.scaling[active_adapter]

            x = x.to(lora_B[0].weight.dtype)

            route_weight = F.softmax(lora_gating(x), dim=-1) # [bsz, seqlen, expert_num]

            for i in range(len(lora_B)):
                if self.hydra:
                    tmp = torch.unsqueeze(route_weight[:,:,i], -1) * lora_B[i](lora_A(dropout(x))) * scaling
                else:
                    tmp = torch.unsqueeze(route_weight[:,:,i], -1) * lora_B[i](lora_A[i](dropout[i](x))) * scaling
                result += tmp

        result = result.to(previous_dtype)
        return result
