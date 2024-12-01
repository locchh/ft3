import os
import sys
import torch
from tqdm import tqdm
from transformers import pipeline
from datasets import load_dataset
from utils.helper import calculate_metrics

# Set up environment for GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Check GPU availability and get GPU information
def check_gpu():
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        print(f"Current GPU: {gpu_name}")
    else:
        print("No GPU available.")

# Load the dataset
def load_and_prepare_dataset(dataset_name, subset_fraction=0.1):
    dataset = load_dataset(dataset_name)
    print(f"Available Splits: {dataset.keys()}")
    
    train_data = dataset["train"]
    train_data = train_data.select(range(int(subset_fraction * len(train_data))))  # Select 10% of the dataset
    
    return train_data

# Initialize text generation pipeline
def create_generation_pipeline(model_id="meta-llama/Llama-3.2-1B-Instruct"):
    return pipeline(
        "text-generation",
        model=model_id,
        device_map="auto",
    )

# Calculate and accumulate metrics for each sample
def accumulate_metrics(train_data, pipe):
    bert_total, bleu_2_total, bleu_4_total, rouge_2_total, rouge_l_total = 0, 0, 0, 0, 0

    for i in tqdm(range(train_data.num_rows)):
        current_sample = train_data[i]
        messages = current_sample["conversations"]
        reference = messages[-1]['content']
        messages = messages[:-1]

        outputs = pipe(messages, max_new_tokens=128)
        candidate = outputs[0]["generated_text"][-1]['content']

        bert_score, bleu_2_score, bleu_4_score, rouge_2_score, rouge_l = calculate_metrics(reference, candidate)

        # Accumulate scores
        bert_total += bert_score
        bleu_2_total += bleu_2_score
        bleu_4_total += bleu_4_score
        rouge_2_total += rouge_2_score
        rouge_l_total += rouge_l

    return bert_total, bleu_2_total, bleu_4_total, rouge_2_total, rouge_l_total

# Compute average metrics
def compute_average_metrics(bert_total, bleu_2_total, bleu_4_total, rouge_2_total, rouge_l_total, num_items):
    bert_avg = bert_total / num_items
    bleu_2_avg = bleu_2_total / num_items
    bleu_4_avg = bleu_4_total / num_items
    rouge_2_avg = rouge_2_total / num_items
    rouge_l_avg = rouge_l_total / num_items

    return bert_avg, bleu_2_avg, bleu_4_avg, rouge_2_avg, rouge_l_avg

# Export metrics to a text file
def export_metrics_to_file(metrics, file_path="metrics.txt"):
    with open(file_path, "w") as f:
        f.write("Average Metrics for Subset:\n")
        f.write(f"BERT: {metrics[0]}\n")
        f.write(f"BLEU-2: {metrics[1]}\n")
        f.write(f"BLEU-4: {metrics[2]}\n")
        f.write(f"ROUGE-2: {metrics[3]}\n")
        f.write(f"ROUGE-L: {metrics[4]}\n")
    print(f"Metrics saved to {file_path}")

def main():

    # Get model_id
    model_id = sys.argv[1]

    # Check GPU availability
    check_gpu()

    # Load dataset and prepare it
    dataset_name = "locchuong/llama-longquan-llm-japanese-dataset-split_10_250"
    train_data = load_and_prepare_dataset(dataset_name)

    # Create text generation pipeline
    pipe = create_generation_pipeline(model_id=model_id)

    # Initialize accumulators for metrics
    bert_total, bleu_2_total, bleu_4_total, rouge_2_total, rouge_l_total = accumulate_metrics(train_data, pipe)

    # Compute average scores
    num_items = train_data.num_rows
    metrics = compute_average_metrics(bert_total, bleu_2_total, bleu_4_total, rouge_2_total, rouge_l_total, num_items)

    # Print and export the average metrics
    print("Average Metrics for Subset:")
    print(f"BERT: {metrics[0]}")
    print(f"BLEU-2: {metrics[1]}")
    print(f"BLEU-4: {metrics[2]}")
    print(f"ROUGE-2: {metrics[3]}")
    print(f"ROUGE-L: {metrics[4]}")

    # Export metrics to file
    export_metrics_to_file(metrics)

if __name__ == "__main__":
    main()
