"""
CoreLoRA Configuration
"""
from dataclasses import dataclass, field

from ..lora import LoraConfig
from ..utils.peft_types import PeftType


@dataclass
class FlyLoraConfig(LoraConfig):
    """
    FlyLORA Configuration
    """
    top_k: int = field(default=8, metadata={
        "help": "The k in top-k gating."})
    bias_u: float = field(default=0.001, metadata={"help": "bias update rate"})
    
    def __post_init__(self):
        self.peft_type = PeftType.FlyLORA
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
