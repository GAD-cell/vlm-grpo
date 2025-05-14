# VLM-GRPO

Vision Language Model training with GRPO and Unsloth

## Overview

This repository provides tools for training unsloth VLMS using GRPO. It includes:

1. A custom trainer (`VLMGRPOTrainer`) that extends the TRL GRPO trainer to support vision inputs and unsloth
2. Patches for the unsloth library to support VLMs GRPO training

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
model,tokenizer = FastVisionModel.from_pretrained("your-model-name from unsloth available VLMs")

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
    processing_class=tokenizer, # MUST put unsloth processor here !
    reward_processing_classes = tokenizer, #Here also
)

# Train the model
trainer.train()
```

See the `examples` directory for more detailed examples.

## Features

- **VLMGRPOTrainer**: A trainer for Vision Language Models from unsloth using GRPO
- **Unsloth Patches**: Patches for the unsloth library to handle errors gracefully during training
- **Easy Integration**: Works with existing TRL and Hugging Face Transformers code
