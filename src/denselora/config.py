"""
DenseLoRA Configuration
"""
from dataclasses import dataclass, field

from ..lora import LoraConfig
from ..utils.peft_types import PeftType


@dataclass
class DenseLoraConfig(LoraConfig):
    """
    DenseLORA Configuration
    """
    def __post_init__(self):
        self.peft_type = PeftType.DenseLORA
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
