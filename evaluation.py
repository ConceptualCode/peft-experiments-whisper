import torch
import time
import psutil
import argparse
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer, WhisperFeatureExtractor
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import evaluate

def get_system_metrics():
    cpu_usage = psutil.cpu_percent()
    gpu_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
    gpu_usage = 0 
    return cpu_usage, gpu_usage, gpu_memory

def get_model_size(model_dir):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(model_dir):
        for file in filenames:
            fp = os.path.join(dirpath, file)
            total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)

def collate_fn(batch):
    input_features = [torch.tensor(b['input_features']) for b in batch]
    labels = [torch.tensor(b['labels']) for b in batch]

    padded_input_features = pad_sequence(input_features, batch_first=True)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    
    batch_dict = {
        "input_features": padded_input_features,
        "labels": padded_labels
    }
    return batch_dict

def prepare_dataset(batch, processor, tokenizer):
    audio_features_list = []
    labels_list = []
    
    for audio in batch["audio"]:
        inputs = processor.feature_extractor(audio["array"], sampling_rate=16000, return_tensors="pt")
        audio_features_list.append(inputs.input_features[0]) 
    
    for transcript in batch["transcript"]:
        labels = tokenizer(transcript, return_tensors="pt", padding="longest").input_ids[0]
        labels_list.append(labels)
    
    batch["input_features"] = audio_features_list
    batch["labels"] = labels_list
    batch["input_length"] = [f.size(0) for f in audio_features_list]
    batch["labels_length"] = [l.size(0) for l in labels_list]

    return batch

def run_evaluation(model_local, model_name_or_path, batch_size=8):

    model = WhisperForConditionalGeneration.from_pretrained(model_local).cuda() 

    model_dir = f"./{model_local}"
    model_size = get_model_size(model_dir)

    tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, language="English", task="transcribe")
    processor = WhisperProcessor.from_pretrained(model_name_or_path, language="English", task="transcribe")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)

    val_dataset = load_dataset("intronhealth/afrispeech-200", 'igbo', split="test", trust_remote_code=True)

    MAX_DURATION_IN_SECONDS = 30.0
    max_input_length = MAX_DURATION_IN_SECONDS * 16000

    def filter_inputs(input_length):
        return 0 < input_length < max_input_length

    max_label_length = model.config.max_length

    def filter_labels(labels_length):
        return labels_length < max_label_length

  
    val_dataset = val_dataset.map(lambda batch: prepare_dataset(batch, processor, tokenizer), remove_columns=val_dataset.column_names, batched=True)
    val_dataset = val_dataset.filter(filter_inputs, input_columns=["input_length"])
    val_dataset = val_dataset.filter(filter_labels, input_columns=["labels_length"])

  
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

    torch.cuda.reset_peak_memory_stats()

    start_cpu_time = time.time()  # CPU time

    start_gpu = torch.cuda.Event(enable_timing=True)
    end_gpu = torch.cuda.Event(enable_timing=True)
    start_gpu.record()  # GPU time

    predictions = []
    references = []

    for batch in val_loader:
        input_features = batch["input_features"].cuda()

        with torch.no_grad():
            generated_ids = model.generate(input_features)

        predicted_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        predictions.extend(predicted_texts)
        references.extend([processor.decode(label, skip_special_tokens=True) for label in batch["labels"]])

    end_cpu_time = time.time()
    end_gpu.record()

    # Wait for GPU event to finish
    torch.cuda.synchronize()

  
    cpu_elapsed_time = end_cpu_time - start_cpu_time 
    gpu_elapsed_time = start_gpu.elapsed_time(end_gpu) / 1000 

    cpu_usage, gpu_usage, gpu_memory = get_system_metrics()

    wer_metric = evaluate.load("wer")
    wer = wer_metric.compute(predictions=predictions, references=references)

    print(f"Word Error Rate (WER): {wer}")
    print(f"Model Size: {model_size:.2f} MB")
    print(f"CPU Time: {cpu_elapsed_time:.2f} seconds")
    print(f"GPU Time: {gpu_elapsed_time:.2f} seconds")
    print(f"CPU Usage: {cpu_usage:.2f}%")
    print(f"GPU Usage: {gpu_usage}%")
    print(f"GPU Memory Allocated: {gpu_memory:.2f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a Whisper model with Igbo Accented English subset")
    parser.add_argument("--model_local", type=str, required=True, help="Path to the locally fine-tuned Whisper model")
    parser.add_argument("--model_name_or_path", type=str, default="openai/whisper-medium", help="Base Whisper model path (default: openai/whisper-medium)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation (default: 8)")

    args = parser.parse_args()

    run_evaluation(args.model_local, args.model_name_or_path, args.batch_size)