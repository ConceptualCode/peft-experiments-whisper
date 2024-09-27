import torch
import os
from utils import evaluate_model, reset_pruned_model

def magnitude_based_pruning(model, pruning_percentage):
    for name, param in model.named_parameters():
        if "weight" in name and param.requires_grad:
            param_abs = param.data.abs().clone()
            num_params_to_prune = int(pruning_percentage * param_abs.numel())
            if num_params_to_prune > 0:
                threshold, _ = torch.topk(param_abs.view(-1), num_params_to_prune, largest=False)
                threshold_value = threshold[-1]
                mask = param_abs > threshold_value
                param.data.mul_(mask.float())

def apply_pruning_and_finetune(model, trainer, original_model_state, sparsity_levels):
    wer_results = []
    for sparsity_level in sparsity_levels:
        print(f"Applying {sparsity_level * 100}% pruning...")
        magnitude_based_pruning(model, pruning_percentage=sparsity_level)
        model = reset_pruned_model(model, original_model_state)
        trainer.train()  # Fine-tune after each pruning step
        save_dir = f"./whisper_pruned_{int(sparsity_level*100)}_percent"
        os.makedirs(save_dir, exist_ok=True)
        wer = evaluate_model(trainer, sparsity_level, save_dir)
        wer_results.append((sparsity_level, wer))
    return wer_results
