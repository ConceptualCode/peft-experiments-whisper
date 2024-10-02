import torch
import evaluate
import argparse
from datasets import load_dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer, WhisperFeatureExtractor
from peft import LoraConfig, get_peft_model
from lora_quantization import apply_lora, quantize_and_finetune_with_lora
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import os


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

    ranks = [4, 8, 16, 32, 64]
    
    for r in ranks:
        print(f"Running quantization and LoRA fine-tuning for rank {r}")

        
        output_dir = f"./whisper-{args.model_size}-lora_rank-{r}-afrispeech-finetune"

    
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            learning_rate=2e-5,
            evaluation_strategy="steps",
            logging_steps=1,
            max_steps=1,
            predict_with_generate=True,
            fp16=False
        )

        loraTrainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=processor.feature_extractor,
        )
        
        save_dir = f"./whisper_{args.model_size}_lora_rank_{r}"
        
        quantized_lora_model = quantize_and_finetune_with_lora(
            model_name_or_path=model_name,
            args=args,
            r_value=r,
            save_dir=save_dir
        )

        loraTrainer.model = quantized_lora_model
        loraTrainer.train()


        model.save_pretrained(f"{output_dir}/model")
        processor.save_pretrained(f"{output_dir}/processor")
        tokenizer.save_pretrained(f"{output_dir}/tokenizer")

if __name__ == "__main__":
    main()