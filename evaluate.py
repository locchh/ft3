import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import random

import torch
# Check if a GPU is available
if torch.cuda.is_available():
    # Get the current device index (default is 0 if no other device is specified)
    current_device = torch.cuda.current_device()
    
    # Get the name of the GPU at this device index
    gpu_name = torch.cuda.get_device_name(current_device)
    print(f"Current GPU: {gpu_name}")
else:
    print("No GPU available.")

from transformers import pipeline
from datasets import load_dataset
from utils.helper import calculate_metrics

print("Import Successfull!")

# Load dataset
dataset_name = "locchuong/llama-longquan-llm-japanese-dataset-split_10_250"
dataset = load_dataset(dataset_name)

# View available dataset splits
print("Available Splits:", dataset.keys())

# Load specific split (e.g., 'train') and inspect the first few rows
train_data = dataset["train"]

# Create pipeline
model_id = "meta-llama/Llama-3.2-1B-Instruct"

pipe = pipeline(
    "text-generation",
    model=model_id,
    device_map="auto",
)

print("Pipeline Created!")

# Select random sample
random_sample = train_data[random.choice(range(train_data.num_rows))]
tag = random_sample["tag"]
messages = random_sample["conversations"]
reference = messages[-1]['content']
messages = messages[:-1]
    
print("Reference: ",reference,"\n")

outputs = pipe(
    messages,
    max_new_tokens=128,
)

candidate = outputs[0]["generated_text"][-1]['content']

print("Model: ", candidate)

bert_score, bleu_2_score, bleu_4_score, rouge_2_score, rouge_l = calculate_metrics(reference, candidate)
print("BERT:", bert_score)
print("BLEU-2:", bleu_2_score)
print("BLEU-4:", bleu_4_score)
print("ROUGE-2:", rouge_2_score)
print("ROUGE-L:", rouge_l)