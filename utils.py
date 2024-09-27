import torch
import matplotlib.pyplot as plt

def evaluate_model(trainer, sparsity_level, save_dir):
    eval_result = trainer.evaluate()
    wer = eval_result['eval_wer']
    print(f"WER at sparsity {sparsity_level * 100}%: {wer:.4f}")
    trainer.save_model(save_dir)
    return wer

def reset_pruned_model(model, original_model_state):
    model.load_state_dict(original_model_state, strict=False)
    print("Model reset after pruning.")
    return model

def plot_sparsity_vs_wer(wer_results):
    sparsities = [x[0] * 100 for x in wer_results]
    wer_values = [x[1] for x in wer_results]
    plt.plot(sparsities, wer_values, marker='o')
    plt.title("WER vs. Sparsity")
    plt.xlabel("Sparsity Level (%)")
    plt.ylabel("WER")
    plt.grid(True)
    plt.show()
