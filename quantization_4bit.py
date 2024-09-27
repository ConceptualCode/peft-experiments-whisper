import os
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, WhisperForConditionalGeneration
def apply_4bit_quantization(model_name):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb_config
    )
    save_dir = f"./{model_name.replace('/', '-')}-4bit-quantized"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    print(f"4-bit quantized model saved at {save_dir}.")
    return model
