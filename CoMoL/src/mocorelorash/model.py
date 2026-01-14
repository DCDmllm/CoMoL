"""
MoLE Model
"""
from typing import Any

import torch
from peft.tuners.tuners_utils import BaseTunerLayer
import torch.nn as nn

import math

from .config import MoCoreLoraShConfig
from .layer import MoCoreLoraLayer, LinearMoCoreLoraLayer
from ..lora import LoraModel

class MoCoreLoraShModel(LoraModel):
    """
    mixture of lora in core space, layer sharing version

    """
    prefix: str = "lora_"

    def __init__(self, model, config, adapter_name="default") -> None:
        super().__init__(model, config, adapter_name)
        
    
    def reset_parameters(self, adapter_name: str="default"):
        nn.init.kaiming_uniform_(self.lora_share_q_A[adapter_name].weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_share_q_B[adapter_name].weight)
        nn.init.kaiming_uniform_(self.lora_share_k_A[adapter_name].weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_share_k_B[adapter_name].weight)
        nn.init.kaiming_uniform_(self.lora_share_v_A[adapter_name].weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_share_v_B[adapter_name].weight)
        nn.init.kaiming_uniform_(self.lora_share_o_A[adapter_name].weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_share_o_B[adapter_name].weight)
        nn.init.kaiming_uniform_(self.lora_share_up_A[adapter_name].weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_share_up_B[adapter_name].weight)
        nn.init.kaiming_uniform_(self.lora_share_down_A[adapter_name].weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_share_down_B[adapter_name].weight)

    def _create_and_replace(
        self, mocorelorash_config: MoCoreLoraShConfig, adapter_name: str,
        target: nn.Module, target_name: str, parent: nn.Module, **kwargs: Any,
    ) -> None:
        """
        Inplace replacement of the target module with the adapter layer
        """
        kwargs = {
            "lora_rank": mocorelorash_config.lora_rank,
            "lora_alpha": mocorelorash_config.lora_alpha,
            "lora_dropout": mocorelorash_config.lora_dropout,
            "init_lora_weights": mocorelorash_config.init_lora_weights,
            "num_experts": mocorelorash_config.num_experts,
            "core_router": mocorelorash_config.core_router,
        }

        if not hasattr(self, "lora_share_q_A"):
            # get corresponding dimension from model
            self.lora_share_q_A = nn.ModuleDict({})
            self.lora_share_q_B = nn.ModuleDict({})
            self.lora_share_k_A = nn.ModuleDict({})
            self.lora_share_k_B = nn.ModuleDict({})
            self.lora_share_v_A = nn.ModuleDict({})
            self.lora_share_v_B = nn.ModuleDict({})
            self.lora_share_o_A = nn.ModuleDict({})
            self.lora_share_o_B = nn.ModuleDict({})
            self.lora_share_up_A = nn.ModuleDict({})
            self.lora_share_up_B = nn.ModuleDict({})
            self.lora_share_down_A = nn.ModuleDict({})
            self.lora_share_down_B = nn.ModuleDict({})
            print(self.model.config) # base_model config
            if self.model.config.architectures[0] == 'Qwen3ForCausalLM':
                if self.model.config.hidden_size == 4096:
                    # Qwen3 8b
                    dim = 4096
                    kv_out_dim = 1024
                    up_out_dim = 12288
                elif self.model.config.hidden_size == 5120:
                    # Qwen3 14b
                    dim = 5120
                    kv_out_dim = 1024
                    up_out_dim = 17408
            elif self.model.config.architectures[0] == 'LlamaForCausalLM':
                if self.model.config.hidden_size == 4096:
                    # llama3.1 8b
                    dim = 4096
                    kv_out_dim = 1024
                    up_out_dim = 14336
            self.lora_share_q_A[adapter_name] = nn.Linear(dim, mocorelorash_config.lora_rank, bias=False)
            self.lora_share_q_B[adapter_name] = nn.Linear(mocorelorash_config.lora_rank, dim, bias=False)
            self.lora_share_k_A[adapter_name] = nn.Linear(dim, mocorelorash_config.lora_rank, bias=False)
            self.lora_share_k_B[adapter_name] = nn.Linear(mocorelorash_config.lora_rank, kv_out_dim, bias=False)
            self.lora_share_v_A[adapter_name] = nn.Linear(dim, mocorelorash_config.lora_rank, bias=False)
            self.lora_share_v_B[adapter_name] = nn.Linear(mocorelorash_config.lora_rank, kv_out_dim, bias=False)
            self.lora_share_o_A[adapter_name] = nn.Linear(dim, mocorelorash_config.lora_rank, bias=False)
            self.lora_share_o_B[adapter_name] = nn.Linear(mocorelorash_config.lora_rank, dim, bias=False)
            self.lora_share_up_A[adapter_name] = nn.Linear(dim, mocorelorash_config.lora_rank, bias=False)
            self.lora_share_up_B[adapter_name] = nn.Linear(mocorelorash_config.lora_rank, up_out_dim, bias=False)
            self.lora_share_down_A[adapter_name] = nn.Linear(up_out_dim, mocorelorash_config.lora_rank, bias=False)
            self.lora_share_down_B[adapter_name] = nn.Linear(mocorelorash_config.lora_rank, dim, bias=False)
            self.reset_parameters()

        # for qwen3
        if target_name == "q_proj":
            kwargs["lora_A"] = self.lora_share_q_A[adapter_name]
            kwargs["lora_B"] = self.lora_share_q_B[adapter_name]
        elif target_name == "k_proj":
            kwargs["lora_A"] = self.lora_share_k_A[adapter_name]
            kwargs["lora_B"] = self.lora_share_k_B[adapter_name]
        elif target_name == "v_proj":
            kwargs["lora_A"] = self.lora_share_v_A[adapter_name]
            kwargs["lora_B"] = self.lora_share_v_B[adapter_name]
        elif target_name == "o_proj":
            kwargs["lora_A"] = self.lora_share_o_A[adapter_name]
            kwargs["lora_B"] = self.lora_share_o_B[adapter_name]
        elif target_name == "up_proj":
            kwargs["lora_A"] = self.lora_share_up_A[adapter_name]
            kwargs["lora_B"] = self.lora_share_up_B[adapter_name]
        elif target_name == "down_proj":
            kwargs["lora_A"] = self.lora_share_down_A[adapter_name]
            kwargs["lora_B"] = self.lora_share_down_B[adapter_name]

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
