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
from vlmgrpo import VLMGRPOTrainer # YOU MUST IMPORT vlmgrpo before unsloth
from trl import GRPOConfig
from unsloth import FastVisionModel
from unsloth import is_bf16_supported

# Load your model
model,tokenizer = FastVisionModel.from_pretrained("your-model-name from unsloth available VLMs")

# Define your reward functions
reward_funcs = [your_reward_function]

# Create the trainer
training_args = GRPOConfig(
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    bf16 = is_bf16_supported(),
    fp16 = not is_bf16_supported(),
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 2, # Decrease if out of memory
    max_prompt_length = 256,
    max_completion_length = 200,
    # num_train_epochs = 1, # Set to 1 for a full training run
    report_to = "none", # Can use Weights & Biases
    output_dir = "outputs",
)
trainer = VLMGRPOTrainer(
    model=model,
    reward_funcs=reward_funcs,
    args=training_args),
    train_dataset=your_train_dataset,
    processing_class=tokenizer, # MUST put unsloth processor here !
    reward_processing_classes = tokenizer, #Here also
    grad_verbose = True #Enable to monitor loss and grad during training 
)

# Train the model
trainer.train()
```

See the `examples` directory for more detailed examples.
## Dataset

The trainer is implemented for a specific input type : 

```python

{
"prompt": [
    {
    "role": "user",
    "content": [
        {"type": "image"}, # N times if you have an image sequence of length N
        {"type": "text",  "text": "Your super prompt"}]
    }]
"image": [a,list,of,images] # len==N,
"answer": "assistant expected answer according to the prompt"
}



```

## Features

- **VLMGRPOTrainer**: A trainer for Vision Language Models from unsloth using GRPO
- **Unsloth Patches**: Patches for the unsloth library to handle VLMs
- **Easy Integration**: Works with existing TRL and Hugging Face Transformers code

## Limitations
- **Videos input**: Doesn't support for now video input, only images or image sequence
- **VLLm** : Still need to add vllm support to the code, will release it soon
- **Tested** : For now i've tested my implementation only with Qwen2 VL and Qwen2.5 VL


## Issues

If you encounter any problems while using this library, please open an issue on GitHub. I am actively maintaining this repo and will address reported issues.


## Training test

![Training test](images/training_test.png)
