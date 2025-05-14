"""
Patches for unsloth library to support VLM-GRPO training
"""

from .unsloth_patch import patch_requires_grad_post_hook, requires_grad_for_gradient_checkpointing

__all__ = ["patch_requires_grad_post_hook", "requires_grad_for_gradient_checkpointing"]