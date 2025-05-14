  # a faire : créer une fonction pour patcher peft_utils
def requires_grad_post_hook(module, input, output):
    type_output = type(output)
    if type_output is torch.Tensor:
        output.requires_grad_(True)
    else:
        try: # For dataclass from HF, try on loss or logits, depends on the module
            if hasattr(output, "loss") and output.loss is not None:
                output.loss.requires_grad_(True)
            elif hasattr(output, "logits") and output.logits is not None:
                output.logits.requires_grad_(True)
            else:
                raise ValueError("Neither loss nor logits are available")
        except Exception as e:
            raise RuntimeError(f"Unsloth: Failed to make output require gradients: {e}")
pass