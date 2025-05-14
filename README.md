# GRPO for Unsloth VLM

## Overview

This repository contains an implementation of GRPO compatible with VLMs from Unsloth.
The implementation extends the standard GRPO trainer from the TRL library of HuggingFace to work with multimodal vision-language models, allowing for training on both text and image inputs. Thus, the code is based on the original implementation of TRL's GRPO trainer.

## Components

### 1. GRPO Trainer for VLM

The `VLMGRPOTrainer` class in `grpo_trainer.py` extends the standard `GRPOTrainer` to handle vision-language models. Key features include:

- Support for processing both image and text inputs
- Handling of multimodal attention mechanisms
- Proper token generation and reward computation for VLMs
- Integration with Unsloth's FastVisionModel for efficient training

### 2. Unsloth Patch

The `unsloth_patch.py` file contains a hook function (`requires_grad_post_hook`) that patches Unsloth's original method. This patch ensures that gradients are required on logits rather than on loss, which is necessary for GRPO to work correctly with Unsloth models. With GRPO, HuggingFace models don't output a dataclass containing a loss (loss=None) but rather output logits directly in a dataclass. The loss is computed afterward based on these logits.

## How It Works

The implementation follows these key steps:

1. Prepares multimodal inputs (images and text) for the VLM
2. Generates completions based on the inputs
3. Computes rewards using the provided reward functions
4. Normalizes rewards and computes advantages
5. Calculates the GRPO loss, which balances between maximizing rewards and minimizing KL divergence from the reference model

## Dependencies

- PyTorch
- Transformers
- TRL (Transformer Reinforcement Learning)
- Unsloth
- Accelerate

## Usage

```python
#écrire un use case
```

## Patching Unsloth

To apply the patch to Unsloth:

Faut que je fasse une fonction de patch pour que ce soit propre

## Future Plans

- Add support for VLLM (Vector Language Model Library)
- Extend support for VLMs in general, not only Unsloth

## Limitations

- Currently only supports Unsloth's implementation of VLMs with images, not videos.
- The patch for requiring gradients on logits is a temporary solution until proper integration is available

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.