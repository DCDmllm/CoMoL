"""
MoLE Model
"""
from typing import Any

import torch
from peft.tuners.tuners_utils import BaseTunerLayer
from torch import nn

from .config import FlyLoraConfig
from .layer import FlyLoraLayer, LinearFlyLoraLayer
from ..lora import LoraModel


class FlyLoraModel(LoraModel):
    """
    Fly lora Model

    """
    prefix: str = "lora_"

    def __init__(self, model, config, adapter_name="default") -> None:
        super().__init__(model, config, adapter_name)

    def _create_and_replace(
        self, flyLora_config: FlyLoraConfig, adapter_name: str,
        target: nn.Module, target_name: str, parent: nn.Module, **kwargs: Any,
    ) -> None:
        """
        Inplace replacement of the target module with the adapter layer
        """
        kwargs = {
            "lora_rank": flyLora_config.lora_rank,
            "lora_alpha": flyLora_config.lora_alpha,
            "lora_dropout": flyLora_config.lora_dropout,
            "init_lora_weights": flyLora_config.init_lora_weights,
            "bias_u": flyLora_config.bias_u,
            "top_k": flyLora_config.top_k,
        }

        if isinstance(target, FlyLoraLayer):
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
            new_module = LinearFlyLoraLayer(base_layer=target, adapter_name=adapter_name, **kwargs)
        else:
            raise ValueError(
                f"The target module `{target}` is not supported. "
                f"Currently, only the following modules are supported: `torch.nn.Linear`.")

        return new_module
