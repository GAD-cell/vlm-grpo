# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections import OrderedDict
from typing import Union, Any
from unsloth import FastVisionModel
import torch 
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import gather
from transformers import Trainer
from trl import GRPOConfig, GRPOTrainer
from trl.trainer.utils import selective_log_softmax
from trl.data_utils import is_conversational, maybe_apply_chat_template, apply_chat_template
from trl.models import unwrap_model_for_generation
from .patches import patch_requires_grad_post_hook, requires_grad_for_gradient_checkpointing

class VLMGRPOTrainer(GRPOTrainer):
    """
    Trainer for Vision Language Models using Generative Reward-Paired Optimization.
    
    This trainer extends the GRPOTrainer to support Vision Language Models by patching
    the unsloth library and handling vision inputs properly.
    """
    def __init__(
        self,
        model,
        reward_funcs,
        args: GRPOConfig = None,
        train_dataset = None,
        eval_dataset = None,
        processing_class = None,
        reward_processing_classes = None,
        callbacks = None,
        peft_config = None,
    ):
        # Patch unsloth 
        requires_grad_for_gradient_checkpointing(model)
        # Avoid flag error
        if hasattr(model, "_flag_for_generation"):
            setattr(model, "_flag_for_generation", False)
        FastVisionModel.for_training(model)

        super().__init__(
            model = model,
            reward_funcs = reward_funcs,
            args = args,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            processing_class = processing_class,
            reward_processing_classes = reward_processing_classes,
            callbacks = callbacks,
            peft_config = peft_config,
        )
        
        # Override GRPOTrainer tokenizer init
        if processing_class is not None:
            self.processing_class = processing_class
          
        # If passing only 1 processor for reward (should be the same as processing_class)
        if type(reward_processing_classes) == type(processing_class):
            self.reward_processing_classes = [reward_processing_classes for i in range(len(reward_funcs))]

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare inputs for the model.
        
        Args:
            inputs: The inputs to prepare
            
        Returns:
            The prepared inputs
        """
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        image = [x["image"] for x in inputs]
        prompts_text = [self.processing_class.apply_chat_template(example["prompt"], add_generation_prompt=True) for example in inputs]
        prompt_inputs = self.processing_class(
            images=image, text=prompts_text, return_tensors="pt", padding=True, add_special_tokens=False)
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        pixel_values, image_grid_thw = prompt_inputs["pixel_values"], prompt_inputs["image_grid_thw"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]

        # Regular generation path
        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)

        # Compute prompt length and extract completion ids
        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw, logits_to_keep
                    )

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [self.processing_class.apply_chat_template(x["text"], add_generation_prompt=True) for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(rewards.mean().item())
        self._metrics["reward_std"].append(std_grouped_rewards.mean().item())

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }
        return output

    def _get_per_token_logps(self, model, input_ids, attention_mask, pixel_values, image_grid_thw, logits_to_keep):
        """
        Get per-token log probabilities.
        
        Args:
            model: The model to use
            input_ids: The input token IDs
            attention_mask: The attention mask
            pixel_values: The pixel values
            image_grid_thw: The image grid THW
            logits_to_keep: The number of logits to keep
            
        Returns:
            The per-token log probabilities
        """
        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
        logits = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, 
                      image_grid_thw=image_grid_thw, logits_to_keep=logits_to_keep + 1).logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

        input_ids = input_ids[:, -logits_to_keep:]
        # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
        # See https://github.com/huggingface/trl/issues/2770
        logits = logits[:, -logits_to_keep:]
        return selective_log_softmax(logits, input_ids)  # compute logprobs for the input tokens

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the loss for the model.
        
        Args:
            model: The model to compute the loss for
            inputs: The inputs to the model
            return_outputs: Whether to return outputs
            num_items_in_batch: The number of items in the batch
            
        Returns:
            The loss
        """
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        pixel_values, image_grid_thw = inputs["pixel_values"], inputs["image_grid_thw"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, pixel_values, image_grid_thw, logits_to_keep)
        # Compute the KL divergence between the model and the reference model
        ref_per_token_logps = inputs["ref_per_token_logps"]
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # x - x.detach() allows for preserving gradients from x
        advantages = inputs["advantages"]
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        return loss