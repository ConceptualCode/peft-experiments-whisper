import torch
import evaluate
import argparse
from datasets import load_dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer, WhisperFeatureExtractor
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from lora_quantization import apply_lora, lora_finetune
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import os
from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model
# from peft import prepare_model_for_int8_training
from transformers import Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper model with pruning, LoRA, and quantization")
    parser.add_argument("--model_size", type=str, default="small", help="Model size to use (small, medium, large)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_name = f"openai/whisper-{args.model_size}"
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device) 
    processor = WhisperProcessor.from_pretrained(model_name, language="English", task="transcribe")
    tokenizer = WhisperTokenizer.from_pretrained(model_name, language="English", task="transcribe")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    metric = evaluate.load("wer")

    train_dataset = load_dataset("intronhealth/afrispeech-200", 'igbo', split="train")
    val_dataset = load_dataset("intronhealth/afrispeech-200", 'igbo', split="test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    MAX_DURATION_IN_SECONDS = 30.0
    max_input_length = MAX_DURATION_IN_SECONDS * 16000

    def filter_inputs(input_length):
        """Filter inputs with zero input length or longer than 30s"""
        return 0 < input_length < max_input_length

    max_label_length = model.config.max_length

    def filter_labels(labels_length):
        """Filter label sequences longer than max length (448)"""
        return labels_length < max_label_length

    def prepare_dataset(batch):

        audio_features_list = []
        labels_list = []
        
        for audio in batch["audio"]:
        
            inputs = feature_extractor(audio["array"], sampling_rate=16000, return_tensors="pt")
            audio_features_list.append(inputs.input_features[0]) 
        
        for transcript in batch["transcript"]:
            labels = tokenizer(transcript, return_tensors="pt", padding="longest").input_ids[0]
            labels_list.append(labels) 
        
        batch["input_features"] = audio_features_list
        batch["labels"] = labels_list

        batch["input_length"] = [f.size(0) for f in audio_features_list]
        batch["labels_length"] = [l.size(0) for l in labels_list]

        return batch

    train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names, batched=True)
    train_dataset = train_dataset.filter(filter_inputs, input_columns=["input_length"])
    train_dataset = train_dataset.filter(filter_labels, input_columns=["labels_length"])

    val_dataset = val_dataset.map(prepare_dataset, remove_columns=val_dataset.column_names, batched=True)
    val_dataset = val_dataset.filter(filter_inputs, input_columns=["input_length"])
    val_dataset = val_dataset.filter(filter_labels, input_columns=["labels_length"])

    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt", padding=True)

            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt", padding=True)

            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]

            batch["labels"] = labels 
            return batch

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # This callback helps to save only the adapter weights and remove the base model weights.
    class SavePeftModelCallback(TrainerCallback):
        def on_save(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ):
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

            peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
            kwargs["model"].save_pretrained(peft_model_path)

            pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
            if os.path.exists(pytorch_model_path):
                os.remove(pytorch_model_path)
            return control

    ranks = [32]

    # model = prepare_model_for_kbit_training(model) # , output_embedding_layer_name="proj_out")
    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)

    # model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)

    # config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")


    
    for r in ranks:
        print(f"Running quantization and LoRA fine-tuning for rank {r}")
        ranks = [32]

        model = prepare_model_for_kbit_training(model) # , output_embedding_layer_name="proj_out")
        model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)

        config = LoraConfig(r=r, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
        
        output_dir = f"./whisper-{args.model_size}-lora_rank-{r}-afrispeech-finetune"
    
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=50,
            learning_rate=1e-3,
            evaluation_strategy="steps",
            logging_steps=50,
            max_steps=1000,
            predict_with_generate=True,
            generation_max_length=128,
            gradient_accumulation_steps=1,
            fp16=True,
            label_names=["labels"],
            max_grad_norm=1.0
        )

        loraTrainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=processor.feature_extractor,
            callbacks=[SavePeftModelCallback]   
        )
        
        model.config.use_cache = False
        
        # save_dir = f"./whisper_{args.model_size}_lora_rank_{r}"
        
        # quantized_lora_model = lora_finetune(
        #     model_name_or_path=model_name,
        #     args=args,
        #     r_value=r,
        #     save_dir=save_dir
        # )

        # loraTrainer.model = quantized_lora_model
        loraTrainer.train()


        # model.save_pretrained(f"{output_dir}/model")
        # processor.save_pretrained(f"{output_dir}/processor")
        # tokenizer.save_pretrained(f"{output_dir}/tokenizer")

if __name__ == "__main__":
    main()