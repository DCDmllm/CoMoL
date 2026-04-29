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
        self.lora_A = nn.ModuleDict({})
        self.lora_Cores = nn.ParameterDict({})
        self.lora_B = nn.ModuleDict({})
        
        self.lora_router = nn.ModuleDict({})

        self.kwargs = kwargs

        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

        self.in_features = in_features
        self.out_features = out_features

    def update_layer(
        self, adapter_name: str, lora_rank: int, lora_alpha: int, lora_dropout: float, init_lora_weights: bool,
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
        self.lora_A[adapter_name] = nn.Linear(self.in_features, lora_rank, bias=False)
        self.lora_Cores[adapter_name] = nn.ParameterList([torch.randn(lora_rank, lora_rank) for _ in range(num_experts)])  # 初始化比较重要
        self.lora_B[adapter_name] = nn.Linear(lora_rank, self.out_features, bias=False)
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
        elif adapter_name in self.lora_A.keys():
            nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
            for i in range(len(self.lora_Cores[adapter_name])):
                nn.init.kaiming_uniform_(self.lora_Cores[adapter_name][i], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[adapter_name].weight)

            # cores = self.lora_Cores[adapter_name]
            # for m in cores:
            #     nn.init.orthogonal_(m, gain=1)
                


class LinearMoCoreLoraLayer(nn.Module, MoCoreLoraLayer):
    """
    MoCoreLoRA Implementation in a Linear Layer
    """

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        lora_rank: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: bool = True,
        num_experts: int = 8,
        core_router: bool = False,
        use_core_loss: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        MoCoreLoraLayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.core_router = core_router
        self.use_core_loss = use_core_loss
        self.update_layer(adapter_name, lora_rank, lora_alpha, lora_dropout, init_lora_weights, num_experts, core_router)

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

    def get_orthogonality_loss(self, lora_cores_list):
        """
        计算核心矩阵之间的正交性约束损失。
        
        Args:
            lora_cores_list: nn.ParameterList, 包含 N 个形状为 (r, r) 的张量
            
        Returns:
            loss: 标量张量
        """
        # 1. 将所有专家矩阵 stack 并展平: (N, r, r) -> (N, r^2)
        # 假设 lora_cores_list 中有 N 个专家，每个专家大小为 r*r
        experts = torch.stack([m for m in lora_cores_list])  # shape: (N, r, r)
        N, r, _ = experts.shape
        flattened_experts = experts.view(N, -1)  # shape: (N, r^2)

        # 2. 计算每个向量的 Frobenius 范数 (即 vec(Mi) 的 L2 范数)
        # 为了数值稳定性，添加一个极小值 eps
        norms = torch.norm(flattened_experts, p=2, dim=1, keepdim=True) + 1e-8
        
        # 3. 归一化，得到单位向量
        normalized_experts = flattened_experts / norms  # shape: (N, r^2)

        # 4. 计算余弦相似度矩阵 (Gram 矩阵): S = M * M^T
        # S[i, j] 即为 Mi 和 Mj 之间的 Tr(Mi^T * Mj) / (|Mi| * |Mj|)
        cosine_sim_matrix = torch.matmul(normalized_experts, normalized_experts.t()) # (N, N)

        # 5. 提取非对角线元素 (即 i != j 的项)
        # 我们希望非对角线元素趋近于 0（即专家之间正交）
        mask = torch.eye(N, device=cosine_sim_matrix.device)
        off_diag_sim = cosine_sim_matrix * (1 - mask)

        # 6. 计算损失：非对角线元素的平方和
        # 按照公式：L_ortho = sum_{i != j} (Similarity_{ij})^2
        loss = torch.sum(off_diag_sim ** 2) / (N * (N - 1)) # 除以专家对数量进行平均

        return loss
    
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
            lora_Cores = self.lora_Cores[active_adapter]
            lora_B = self.lora_B[active_adapter]
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
            # result += lora_B((lora_A_x.unsqueeze(-2) @ lora_Core).squeeze(-2)) * scaling
            result += lora_B(dropout(lora_A_x.unsqueeze(-2) @ dropout(lora_Core)).squeeze(-2)) * scaling
            
            # router_logits = F.softmax(lora_router(x), dim=-1)
            # # fuse cores
            # lora_Core = torch.sum(torch.stack([core * router_logits[:, :, i].unsqueeze(-1).unsqueeze(-1) for i, core in enumerate(lora_Cores)]), dim=0)
            # # print('lora_Core.shape', lora_Core.shape)
            # result += lora_B((lora_A(dropout(x)).unsqueeze(-2) @ lora_Core).squeeze(-2)) * scaling
            if self.use_core_loss and self.training:
                self.core_loss = self.get_orthogonality_loss(lora_Cores)
                # print(self.core_loss)

        result = result.to(previous_dtype)
        return result
