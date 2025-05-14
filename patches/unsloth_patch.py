from importlib import import_module
import functools
import torch
import inspect
import re
from collections import OrderedDict

def patch_requires_grad_post_hook():
    """
    Patches the requires_grad_post_hook function in unsloth-zoo.peft_utils
    """
    try:
        peft_utils = import_module('unsloth_zoo.peft_utils')
        
        original_function = peft_utils.requires_grad_for_gradient_checkpointing
        
        @functools.wraps(original_function) 
        def wrapper(model):
            try:
                return original_function(model)
            except Exception as e:
                
                return requires_grad_for_gradient_checkpointing(model)
        
        peft_utils.requires_grad_for_gradient_checkpointing = wrapper
        
        print("Unsloth patched for VLMs GRPO training")
        return True
    except Exception as e:
        raise RuntimeError(f"Failed to patch requires_grad_post_hook: {str(e)}")

# Original code from unsloth but with the modified version of requires_grad_post_hook
def requires_grad_for_gradient_checkpointing(model):
    # All Unsloth Zoo code licensed under LGPLv3
    # Enables requires_grad to make gradient checkpointing work on
    # non language models that don't just use .embed_tokens
    def register_other_hooks(name1, name2, module, _hooks):
        old_hooks = eval(f"module.{_hooks}")
        other_hooks = []
        for value in old_hooks.values():
            qualname = getattr(value, "__qualname__", "")
            name     = getattr(value, "__name__", "")
            if name1 in qualname or name2 in qualname: pass
            elif name2 in name or name2 in name: pass
            else: other_hooks.append(value)
        pass
        # Keep none input requires grad hooks
        exec(f"module.{_hooks} = OrderedDict()")
        for hook in other_hooks:
            exec(f"module.register{_hooks[:-1]}(hook)")
        pass
    pass

    # Remove all previous forward hooks for gradient checkpointing
    for name, module in model.named_modules():
        if len(module._forward_hooks) != 0:
            register_other_hooks(
                "enable_input_require_grads",
                "make_inputs_require_grad",
                module,
                "_forward_hooks",
            )
        pass
    pass

    # Add post forward hook
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

    def requires_grad_pre_hook(module, input):
        type_input = type(input)
        if type_input is torch.Tensor:
            input.requires_grad_(True)
        elif type_input is tuple or type_input is list:
            if len(input) == 0:
                raise RuntimeError("Unsloth: Failed to make input require gradients!")
                # print(f"  WARNING: Empty list input to {module.__class__.__name__}!") # 
                # return
            if torch.is_floating_point(input[0]):
                input[0].requires_grad_(True)
        else:
            raise RuntimeError("Unsloth: Failed to make input require gradients!")
    pass

    # Find 1st ever item which requires grad
    param = None
    for name, param in model.named_parameters():
        if param.requires_grad: break
    if param is None: return

    name = re.sub("\.([\d]{1,})\.", r"[\1].", name)
    name_components = name.split(".")

    if len(name_components) == 0:
        raise RuntimeError("Unsloth: Model has 0 layers?")

    final_where = None
    # Try getting previous parent module
    for j in range(len(name_components)-1, 0, -1):
        name_curr = name_components[j]
        name_pre  = "model." + ".".join(name_components[:j])
        # Disable [\d] since it fails in gradient checkpointing
        if re.search(r"\[[\d]{1,}\]", name_pre): continue
        module = eval(name_pre)
        if hasattr(module, "forward"):
            try: forward = inspect.getsource(module.forward)
            except: continue

            # Normal self.language_model(...)
            if f"self.{name_curr}(" in forward:
                final_where = j + 1
                break

            # Fix self.blocks[0] like in Qwen
            module_list = re.sub(r"\[[\d]{1,}\]", "", name_curr)
            if f"in self.{module_list}:" in forward:
                final_where = j
                break
            elif re.search(r"for [^\s]{3,} in self\." + module_list, forward) is not None:
                # Might have failed finding self.layers: like self.layers[...]:
                final_where = j
                break
            pass
        pass
    pass

    if final_where is None:
        # Find all input embeddings and just set them all as a fallback!
        # Add other hooks first
        register_other_hooks(
            "requires_grad_post_hook",
            "requires_grad_post_hook",
            module,
            "_forward_hooks",
        )
        module.register_forward_hook(requires_grad_post_hook)
        return
    pass

    module_name = "model." + ".".join(name_components[:final_where])
    module = eval(module_name)

    if hasattr(module, "config") and (module.config.__class__.__name__ in ("CLIPVisionConfig", "SiglipVisionConfig",)):
        # CLIP - backtrack to get_input_embeddings since requires_grad fails!
        old_module = model
        for module_name, module in model.named_modules():
            if not hasattr(module, "get_input_embeddings"): break
            old_module = module
        module = old_module
    pass
    print(f"Unsloth: Making `{module_name}` require gradients")

    still_need_patching = True
    # Check if input_embeddings exists
    if hasattr(module, "get_input_embeddings"):
        # Use forward hook after Embedding() is called
        try:
            module = module.get_input_embeddings()
            # Add other hooks first
            register_other_hooks(
                "requires_grad_post_hook",
                "requires_grad_post_hook",
                module,
                "_forward_hooks",
            )
            module.register_forward_hook(requires_grad_post_hook)
            still_need_patching = False
        except:
            # Not Implemented probably?
            still_need_patching = True
    pass

    if still_need_patching:
        # Use forward pre hook before module is called
        register_other_hooks(
            "requires_grad_pre_hook",
            "requires_grad_pre_hook",
            module,
            "_forward_pre_hooks",
        )
        module.register_forward_pre_hook(requires_grad_pre_hook)
    pass
pass