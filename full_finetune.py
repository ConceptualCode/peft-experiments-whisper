import torch
import evaluate
import argparse
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments

# Argument Parser for model size
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper model with dynamic model size selection")
    parser.add_argument("--model_size", type=str, default="small", help="Model size to use (small, medium, large)")
    return parser.parse_args()

# Main function
def main():
    args = parse_args()
    
    # Load Whisper model, processor, and tokenizer based on model size
    model_name = f"openai/whisper-{args.model_size}"
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    processor = WhisperProcessor.from_pretrained(model_name, language="English", task="transcribe")
    tokenizer = WhisperTokenizer.from_pretrained(model_name, language="English", task="transcribe")
    metric = evaluate.load("wer")

    # Load dataset
    train_dataset = load_dataset("intronhealth/afrispeech-200", 'igbo', split="train")
    val_dataset = load_dataset("intronhealth/afrispeech-200", 'igbo', split="test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Data preparation
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
            # Extract the features for each audio sample
            inputs = processor.feature_extractor(audio["array"], sampling_rate=16000, return_tensors="pt")
            audio_features_list.append(inputs.input_features[0]) 
        
        # Tokenize each transcript in the batch
        for transcript in batch["transcript"]:
            labels = tokenizer(transcript, return_tensors="pt", padding="longest").input_ids[0]
            labels_list.append(labels) 
        
        # Store the processed features and labels back into the batch
        batch["input_features"] = audio_features_list
        batch["labels"] = labels_list

        # Compute input length and label length for filtering purposes
        batch["input_length"] = [f.size(0) for f in audio_features_list]
        batch["labels_length"] = [l.size(0) for l in labels_list]

        return batch

    # Preprocess datasets
    train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names, batched=True)
    train_dataset = train_dataset.filter(filter_inputs, input_columns=["input_length"])
    train_dataset = train_dataset.filter(filter_labels, input_columns=["labels_length"])

    val_dataset = val_dataset.map(prepare_dataset, remove_columns=val_dataset.column_names, batched=True)
    val_dataset = val_dataset.filter(filter_inputs, input_columns=["input_length"])
    val_dataset = val_dataset.filter(filter_labels, input_columns=["labels_length"])

    # Data collator with padding
    from dataclasses import dataclass
    from typing import Any, Dict, List, Union

    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt", padding=True)

            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]

            batch["labels"] = labels
            return batch

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Metric
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        label_ids[label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"./whisper-{args.model_size}-finetune",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        logging_dir='./logs',
        logging_steps=50,
        learning_rate=2e-5,
        evaluation_strategy="steps",
        max_steps=1,
        #save_steps=1000,
        eval_steps=1,
        save_total_limit=2,
        num_train_epochs=3,
        predict_with_generate=True,
        fp16=True 
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model, 
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset, 
        data_collator=data_collator,
        compute_metrics=compute_metrics, 
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    model.save_pretrained(f"./whisper-{args.model_size}-finetuned")
    processor.save_pretrained(f"./whisper-{args.model_size}-finetuned")
    trainer.evaluate()


if __name__ == "__main__":
    main()
