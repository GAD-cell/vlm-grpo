"""
VLM-GRPO: Vision Language Model training with Generative Reward-Paired Optimization
"""

from .trainer import VLMGRPOTrainer
from .patches import patch_requires_grad_post_hook

__all__ = ["VLMGRPOTrainer", "patch_requires_grad_post_hook"]