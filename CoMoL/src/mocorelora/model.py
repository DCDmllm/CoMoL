"""
MoLE Model
"""
from typing import Any

import torch
from peft.tuners.tuners_utils import BaseTunerLayer
from torch import nn

from .config import MoCoreLoraConfig
from .layer import MoCoreLoraLayer, LinearMoCoreLoraLayer
from ..lora import LoraModel


class MoCoreLoraModel(LoraModel):
    """
    Core lora Model

    """
    prefix: str = "lora_"

    def __init__(self, model, config, adapter_name="default") -> None:
        super().__init__(model, config, adapter_name)

    def _create_and_replace(
        self, mocoreLora_config: MoCoreLoraConfig, adapter_name: str,
        target: nn.Module, target_name: str, parent: nn.Module, **kwargs: Any,
    ) -> None:
        """
        Inplace replacement of the target module with the adapter layer
        """
        kwargs = {
            "lora_rank": mocoreLora_config.lora_rank,
            "lora_alpha": mocoreLora_config.lora_alpha,
            "lora_dropout": mocoreLora_config.lora_dropout,
            "init_lora_weights": mocoreLora_config.init_lora_weights,
            "num_experts": mocoreLora_config.num_experts,
            "core_router": mocoreLora_config.core_router,
        }

        if isinstance(target, MoCoreLoraLayer):
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
            new_module = LinearMoCoreLoraLayer(base_layer=target, adapter_name=adapter_name, **kwargs)
        else:
            raise ValueError(
                f"The target module `{target}` is not supported. "
                f"Currently, only the following modules are supported: `torch.nn.Linear`.")

        return new_module
