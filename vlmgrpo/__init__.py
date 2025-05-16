"""
VLM-GRPO: Vision Language Model training with Generative Reward-Paired Optimization
"""

from .patches import patch_requires_grad_post_hook
import os
if "UNSLOTH_IS_PRESENT" in os.environ:
    raise RuntimeError(
        "Unsloth is already present in the environment. "
        "Please import vlmgrpo before importing unsloth."
    )
else : 
    os.environ["UNSLOTH_IS_PRESENT"] = "1"
    import unsloth_zoo.peft_utils

patch_requires_grad_post_hook()
from .trainer import VLMGRPOTrainer

__all__ = ["VLMGRPOTrainer", "patch_requires_grad_post_hook"]