import os
import torch
import logging
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
from datasets import load_dataset


def setup_logging(output_dir: str) -> logging.Logger:
    """
    Set up logging to both a file and the console.

    Args:
        output_dir (str): Directory where logs will be saved.

    Returns:
        logging.Logger: Configured logger instance.
    """
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
    return logging.getLogger(__name__)


class LogEvalLossCallback(TrainerCallback):
    """
    Custom callback for logging evaluation loss during training.
    """
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        self.logger.info(f"Evaluation loss: {metrics.get('eval_loss', 'N/A')}")


def configure_device(logger: logging.Logger) -> None:
    """
    Configure the device settings for PyTorch.

    Args:
        logger (logging.Logger): Logger instance for logging device info.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        logger.info(f"Using GPU: {gpu_name}")
    else:
        logger.warning("No GPU available. Falling back to CPU.")


def preprocess_dataset(dataset_name: str, subset_fraction: float, tokenizer, logger: logging.Logger):
    """
    Load and preprocess the dataset.

    Args:
        dataset_name (str): Name of the dataset to load.
        subset_fraction (float): Fraction of the dataset to use.
        tokenizer: Tokenizer for processing text.
        logger (logging.Logger): Logger for logging messages.

    Returns:
        Dataset: Tokenized and processed dataset split into training and test sets.
    """
    logger.info(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")
    dataset = dataset.filter(lambda x: len(x['conversations']) > 0)
    dataset = dataset.select(range(int(subset_fraction * len(dataset))))
    logger.info(f"Dataset reduced to {len(dataset)} samples.")

    def apply_chat_template(example):
        messages = example["conversations"]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return {"prompt": prompt}

    dataset = dataset.map(apply_chat_template)
    dataset = dataset.train_test_split(test_size=0.1)
    logger.info("Dataset split into training and test sets.")

    def tokenize_function(example):
        tokens = tokenizer(
            example['prompt'], 
            padding="max_length", 
            truncation=True, 
            max_length=128
        )
        tokens['labels'] = [
            -100 if token == tokenizer.pad_token_id else token 
            for token in tokens['input_ids']
        ]
        return tokens

    tokenized_dataset = dataset.map(tokenize_function)
    tokenized_dataset = tokenized_dataset.remove_columns(['conversations', 'tag', 'prompt'])
    logger.info("Dataset tokenized and prepared.")
    return tokenized_dataset


def main():
    output_dir = "output"
    logger = setup_logging(output_dir)
    logger.info("Starting script...")

    # Configure device
    configure_device(logger)

    # Load model and tokenizer
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    logger.info(f"Loading model and tokenizer: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    # Preprocess dataset
    dataset_name = "locchuong/llama_conversations_1k"
    subset_fraction = 0.1
    tokenized_dataset = preprocess_dataset(dataset_name, subset_fraction, tokenizer, logger)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=20,
        logging_steps=20,
        save_steps=90,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=2,
        fp16=True,
        log_level="info",
        learning_rate=5e-6,
        max_grad_norm=1.0,
        logging_dir=output_dir,
        report_to=["tensorboard"],
    )

    # Initialize Trainer
    logger.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        callbacks=[LogEvalLossCallback(logger)],
    )

    # Train the model
    logger.info("Starting training...")
    trainer.train()

    # Save the model and tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model and tokenizer saved to {output_dir}")


if __name__ == "__main__":
    main()
