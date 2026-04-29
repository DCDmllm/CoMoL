"""
Configure and Model Mappings

Portions of this file are modifications based on work created and
shared by the HuggingFace Inc. team and used according to terms
described in the Apache License 2.0.
"""

from .adamole import AdaMoleConfig, AdaMoleModel
from .lora import LoraConfig, LoraModel
from .mole import MoleConfig, MoleModel
from .corelora import CoreLoraConfig, CoreLoraModel
from .mocorelora import MoCoreLoraConfig, MoCoreLoraModel
from .mocorelorash import MoCoreLoraShConfig, MoCoreLoraShModel
from .denselora import DenseLoraConfig, DenseLoraModel
from .molora import MoLoraConfig, MoLoraModel
from .flylora import FlyLoraConfig, FlyLoraModel
from .utils.peft_types import PeftType

PEFT_TYPE_TO_CONFIG_MAPPING = {
    PeftType.LORA: LoraConfig,
    PeftType.MOLE: MoleConfig,
    PeftType.ADAMOLE: AdaMoleConfig,
    PeftType.CoreLORA: CoreLoraConfig,
    PeftType.MoCoreLORA: MoCoreLoraConfig,
    PeftType.MoCoreLORASh: MoCoreLoraShConfig,
    PeftType.DenseLORA: DenseLoraConfig,
    PeftType.MoLORA: MoLoraConfig,
    PeftType.FlyLORA: FlyLoraConfig,
}
PEFT_TYPE_TO_MODEL_MAPPING = {
    PeftType.LORA: LoraModel,
    PeftType.MOLE: MoleModel,
    PeftType.ADAMOLE: AdaMoleModel,
    PeftType.CoreLORA: CoreLoraModel,
    PeftType.MoCoreLORA: MoCoreLoraModel,
    PeftType.MoCoreLORASh: MoCoreLoraShModel,
    PeftType.DenseLORA: DenseLoraModel,
    PeftType.MoLORA: MoLoraModel,
    PeftType.FlyLORA: FlyLoraModel,
}
