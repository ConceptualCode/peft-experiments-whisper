import torch
from peft import LoraConfig, get_peft_model
import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import WhisperForConditionalGeneration


def apply_lora(model, lora_config, save_dir):
    
    model_with_lora = get_peft_model(model, lora_config)
    print("LoRA applied.")
    os.makedirs(save_dir, exist_ok=True)
    model_with_lora.save_pretrained(save_dir)
    return model_with_lora


def quantize_and_finetune_with_lora(model_name_or_path, args, r_value, save_dir):
    """
    This function applies 8-bit quantization to a model and then fine-tunes it using LoRA.
    :param model_name_or_path: The name or path of the Whisper model.
    :param args: The arguments passed to the model for configuration.
    :param r_value: The rank value for LoRA configuration.
    :param save_dir: The directory to save the fine-tuned model.
    """

    # Load Whisper model with 8-bit quantization
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name_or_path,
        load_in_8bit=True,
    )
    
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    model = prepare_model_for_kbit_training(model)
    
    # Set LoRA configuration
    lora_config = LoraConfig(
        r=r_value,
        lora_alpha=32,
        target_modules=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj", 
                        "encoder_attn.q_proj", "encoder_attn.k_proj", "encoder_attn.v_proj", "encoder_attn.out_proj"],
        lora_dropout=0.1, 
        bias="none"
    )
    
    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    
    model.print_trainable_parameters()
    
    model.save_pretrained(save_dir)

    return model

