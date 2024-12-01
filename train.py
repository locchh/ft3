import os
import torch
import logging
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
from datasets import load_dataset

print("Import Successful!")

# Set up logging
def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "training.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    return logger

# Callback for logging evaluation loss
class LogEvalLossCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        logger.info(f"Evaluation loss: {metrics['eval_loss']}")

global logger
output_dir = "output"
logger = setup_logging(output_dir)

# Set up GPU if available
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if torch.cuda.is_available():
    current_device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(current_device)
    logger.info(f"Current GPU: {gpu_name}")
else:
    logger.warning("No GPU available.")

# Load the model and tokenizer
model_id = "meta-llama/Llama-3.2-1B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

# Load the dataset
dataset_name = "locchuong/llama_conversations_1k"
subset_fraction = 0.1
dataset = load_dataset(dataset_name, split="train")
# Filter the dataset to exclude rows where 'conversations' length is zero
dataset = dataset.filter(lambda x: len(x['conversations']) > 0)
dataset = dataset.select(range(int(subset_fraction * len(dataset))))  # Select 10% of the dataset

# Apply the chat template function
def apply_chat_template(example):
    messages = example["conversations"]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return {"prompt": prompt}

dataset = dataset.map(apply_chat_template)
dataset = dataset.train_test_split(test_size=0.1)

# Tokenize the data
def tokenize_function(example):
    #tokens = tokenizer(example['prompt'], padding="max_length", truncation=True) # ignore max_length
    tokens = tokenizer(example['prompt'], padding="max_length", truncation=True, max_length=128) # set max_length to avoid out of memory
    tokens['labels'] = [-100 if token == tokenizer.pad_token_id else token for token in tokens['input_ids']]
    return tokens

tokenized_dataset = dataset.map(tokenize_function)
tokenized_dataset = tokenized_dataset.remove_columns(['conversations', 'tag', 'prompt'])


# Set up training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="steps",
    eval_steps=1,
    logging_steps=1,
    save_steps=5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=10,
    fp16=True,
    report_to="none",
    log_level="info",
    learning_rate=0.000005,
    max_grad_norm=1,
    logging_dir=output_dir
)


# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    callbacks=[LogEvalLossCallback()]  # Register the callback
)


# Train the model
logger.info("Starting training...")
trainer.train()

# Save the model and tokenizer
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
logger.info(f"Model and tokenizer saved to {output_dir}")