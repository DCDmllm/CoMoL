"""
MoLE Model
"""
from typing import Any

import torch
from peft.tuners.tuners_utils import BaseTunerLayer
from torch import nn

from .config import MoLoraConfig
from .layer import MoLoraLayer, LinearMoLoraLayer
from ..lora import LoraModel


class MoLoraModel(LoraModel):
    """
    MoLora (Mixture of LoRA Experts) Model
    & 
    HydraLora
    """
    prefix: str = "lora_"

    def __init__(self, model, config, adapter_name="default") -> None:
        super().__init__(model, config, adapter_name)

    def _create_and_replace(
        self, molora_config: MoLoraConfig, adapter_name: str,
        target: nn.Module, target_name: str, parent: nn.Module, **kwargs: Any,
    ) -> None:
        """
        Inplace replacement of the target module with the adapter layer
        """
        kwargs = {
            "lora_rank": molora_config.lora_rank,
            "lora_alpha": molora_config.lora_alpha,
            "lora_dropout": molora_config.lora_dropout,
            "init_lora_weights": molora_config.init_lora_weights,
            "num_experts": molora_config.num_experts,
            "hydra": molora_config.hydra,
        }

        if isinstance(target, MoLoraLayer):
            target.update_layer(adapter_name, **kwargs)
        else:
            new_module = self._create_new_module(adapter_name, target, **kwargs)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(adapter_name: str, target: nn.Module, **kwargs: Any) -> nn.Module:
        """
        Create the new LoRA module for the target module
        """
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            new_module = LinearMoLoraLayer(base_layer=target, adapter_name=adapter_name, **kwargs)
        else:
            raise ValueError(
                f"The target module `{target}` is not supported. "
                f"Currently, only the following modules are supported: `torch.nn.Linear`.")

        return new_module
