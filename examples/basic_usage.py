"""
Basic usage example for the vlmgrpo package.
"""

import torch
from transformers import AutoProcessor
from trl import GRPOConfig
from unsloth import FastVisionModel
from vlmgrpo import VLMGRPOTrainer

def main():
    """
    Example of how to use the VLMGRPOTrainer.
    """
    # Load your model
    model_name = "your-model-name"  # Replace with your model name
    model = FastVisionModel.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    
    # Define a simple reward function
    def simple_reward_function(prompts, completions, **kwargs):
        """
        A simple reward function that rewards longer completions.
        
        Args:
            prompts: The prompts
            completions: The completions
            
        Returns:
            The rewards
        """
        return [len(completion) for completion in completions]
    
    # Create a dummy dataset
    class DummyDataset(torch.utils.data.Dataset):
        """
        A dummy dataset for demonstration purposes.
        """
        def __init__(self, size=100):
            self.size = size
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            return {
                "prompt": [{"role": "user", "content": "Describe this image"}],
                "image": torch.rand(3, 224, 224),  # Random image
            }
    
    # Create the trainer
    trainer = VLMGRPOTrainer(
        model=model,
        reward_funcs=[simple_reward_function],
        args=GRPOConfig(
            output_dir="./output",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            logging_steps=10,
            save_steps=100,
            learning_rate=5e-5,
        ),
        train_dataset=DummyDataset(size=100),
        processing_class=processor,
    )
    
    # Train the model
    trainer.train()
    
    # Save the model
    trainer.save_model("./output/final_model")
    
if __name__ == "__main__":
    main()