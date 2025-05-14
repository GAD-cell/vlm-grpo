# VLM-GRPO

Vision Language Model training with Generative Reward-Paired Optimization

## Overview

This repository provides tools for training Vision Language Models (VLMs) using Generative Reward-Paired Optimization (GRPO). It includes:

1. A custom trainer (`VLMGRPOTrainer`) that extends the TRL GRPO trainer to support vision inputs
2. Patches for the unsloth library to handle errors gracefully during training

## Installation

You can install the package directly from the repository:

```bash
pip install -e .
```

## Usage

Here's a basic example of how to use the VLMGRPOTrainer:

```python
from vlmgrpo import VLMGRPOTrainer
from trl import GRPOConfig
from unsloth import FastVisionModel

# Load your model
model = FastVisionModel.from_pretrained("your-model-name")

# Define your reward functions
reward_funcs = [your_reward_function]

# Create the trainer
trainer = VLMGRPOTrainer(
    model=model,
    reward_funcs=reward_funcs,
    args=GRPOConfig(
        output_dir="./output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
    ),
    train_dataset=your_train_dataset,
    processing_class=your_processing_class,
)

# Train the model
trainer.train()
```

See the `examples` directory for more detailed examples.

## Features

- **VLMGRPOTrainer**: A trainer for Vision Language Models using GRPO
- **Unsloth Patches**: Patches for the unsloth library to handle errors gracefully during training
- **Easy Integration**: Works with existing TRL and Hugging Face Transformers code

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- TRL 0.7+
- Accelerate 0.20+
- Unsloth 0.3+

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.