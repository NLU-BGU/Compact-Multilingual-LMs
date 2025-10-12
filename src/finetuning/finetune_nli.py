"""
finetune_nli.py
Fine-tuning script for Natural Language Inference (NLI) tasks

Supports multiple model architectures:
- BabyBERTa / RoBERTa (sequence classification)
- T5 (text-to-text generation)
- LTG-BERT (sequence classification)

Supports multiple NLI datasets:
- XNLI (English and French)
- ANLI (Adversarial NLI)
- MultiNLI
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
import numpy as np
from tqdm.auto import tqdm

from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    T5ForConditionalGeneration,
    T5TokenizerFast,
    Trainer,
    TrainingArguments,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
import evaluate

# Import utilities
from finetune_utils import (
    load_tokenizer,
    load_model_for_sequence_classification,
    get_model_output_path,
    get_dataset_paths,
    get_default_training_args,
    ensure_dir,
)

# Disable wandb
os.environ["WANDB_DISABLED"] = "true"

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

# NLI label mappings
LABEL_TO_ID = {"entailment": 0, "neutral": 1, "contradiction": 2}
ID_TO_LABEL = {0: "entailment", 1: "neutral", 2: "contradiction"}

# T5 text-to-text labels
T5_LABEL_MAP = {
    0: "entailment",
    1: "neutral", 
    2: "contradiction"
}


# ============================================================================
# Dataset Loading
# ============================================================================

def load_nli_dataset(
    dataset_name: str,
    base_data_dir: str = "/sise/eliorsu-group/lielbin/Research/datasets/data-finetune"
) -> DatasetDict:
    """
    Load NLI dataset by name.
    
    Args:
        dataset_name: Name of dataset (xnli-en, xnli-fr, anli, mnli)
        base_data_dir: Base directory containing datasets
        
    Returns:
        DatasetDict with train, validation, and optionally test splits
    """
    logger.info(f"Loading dataset: {dataset_name}")
    
    paths = get_dataset_paths(dataset_name, base_data_dir)
    
    # Determine which splits are available
    splits = {}
    if 'train' in paths and os.path.exists(paths['train']):
        splits['train'] = paths['train']
    if 'validation' in paths and os.path.exists(paths['validation']):
        splits['validation'] = paths['validation']
    if 'test' in paths and os.path.exists(paths['test']):
        splits['test'] = paths['test']
    
    if not splits:
        raise FileNotFoundError(f"No data files found for {dataset_name}")
    
    # Load dataset
    raw_datasets = load_dataset('json', data_files=splits)
    
    # Log dataset sizes
    for split, dataset in raw_datasets.items():
        logger.info(f"Loaded {len(dataset)} {split} examples")
    
    return raw_datasets


# ============================================================================
# Classification-based NLI (BabyBERTa, RoBERTa, BERT)
# ============================================================================

def preprocess_classification_nli(
    examples: Dict,
    tokenizer: AutoTokenizer,
    max_seq_length: int = 128
) -> Dict:
    """
    Preprocess examples for classification-based NLI.
    
    Args:
        examples: Batch of examples with 'premise', 'hypothesis', 'label'
        tokenizer: Tokenizer to use
        max_seq_length: Maximum sequence length
        
    Returns:
        Preprocessed examples with tokenized inputs
    """
    # Tokenize premise-hypothesis pairs
    encoded = tokenizer(
        examples["premise"],
        examples["hypothesis"],
        truncation=True,
        padding="max_length",
        max_length=max_seq_length,
    )
    
    # Add labels
    encoded["labels"] = examples["label"]
    
    return encoded


def compute_classification_metrics(eval_pred) -> Dict:
    """
    Compute metrics for classification-based NLI.
    
    Args:
        eval_pred: Tuple of (predictions, labels)
        
    Returns:
        Dictionary with accuracy metric
    """
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    
    # Calculate accuracy
    accuracy = (preds == labels).mean()
    
    # Calculate per-class accuracy
    per_class_acc = {}
    for label_id, label_name in ID_TO_LABEL.items():
        mask = labels == label_id
        if mask.sum() > 0:
            per_class_acc[f"accuracy_{label_name}"] = (preds[mask] == labels[mask]).mean()
    
    metrics = {"accuracy": accuracy}
    metrics.update(per_class_acc)
    
    return metrics


def train_classification_nli(
    model_checkpoint: str,
    dataset_name: str,
    output_dir: str,
    max_seq_length: int = 128,
    batch_size: int = 16,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    seed: int = 42
) -> Dict:
    """
    Train classification-based NLI model (BabyBERTa, RoBERTa, BERT).
    
    Args:
        model_checkpoint: Path to pretrained model
        dataset_name: Name of NLI dataset
        output_dir: Directory to save fine-tuned model
        max_seq_length: Maximum sequence length
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        seed: Random seed
        
    Returns:
        Dictionary with evaluation metrics
    """
    set_seed(seed)
    
    logger.info("="*80)
    logger.info("Classification-based NLI Fine-tuning")
    logger.info(f"Model: {model_checkpoint}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Output: {output_dir}")
    logger.info("="*80)
    
    # Load tokenizer and dataset
    tokenizer = load_tokenizer(model_checkpoint)
    raw_datasets = load_nli_dataset(dataset_name)
    
    # Preprocess datasets
    logger.info("Preprocessing datasets...")
    tokenized_datasets = raw_datasets.map(
        lambda x: preprocess_classification_nli(x, tokenizer, max_seq_length),
        batched=True
    )
    
    # Set format for PyTorch
    tokenized_datasets.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )
    
    # Rename 'label' to 'labels' for Trainer
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    
    # Load model
    model = load_model_for_sequence_classification(
        model_checkpoint,
        num_labels=3  # entailment, neutral, contradiction
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        logging_steps=100,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        report_to=[],
        seed=seed,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_classification_metrics,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Evaluate
    logger.info("Evaluating model...")
    eval_results = trainer.evaluate()
    
    # Save evaluation results
    logger.info(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
    
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    # Also save human-readable results
    with open(os.path.join(output_dir, "results.txt"), 'w') as f:
        f.write("NLI Classification Results\n")
        f.write("="*50 + "\n")
        for key, value in eval_results.items():
            f.write(f"{key}: {value:.4f}\n")
    
    logger.info("Training completed successfully!")
    return eval_results


# ============================================================================
# Generative NLI (T5)
# ============================================================================

def preprocess_generative_nli(
    examples: Dict,
    tokenizer: T5TokenizerFast,
    max_input_length: int = 256,
    max_target_length: int = 16
) -> Dict:
    """
    Preprocess examples for generative NLI (T5).
    
    Args:
        examples: Batch of examples with 'premise', 'hypothesis', 'label'
        tokenizer: T5 tokenizer
        max_input_length: Maximum input length
        max_target_length: Maximum target length
        
    Returns:
        Preprocessed examples in T5 format
    """
    # Create input in T5 format: "nli premise: X hypothesis: Y"
    inputs = [
        f"nli premise: {premise} hypothesis: {hypothesis}"
        for premise, hypothesis in zip(examples["premise"], examples["hypothesis"])
    ]
    
    # Convert label IDs to text
    targets = [T5_LABEL_MAP[label] for label in examples["label"]]
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True,
        padding="max_length"
    )
    
    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=max_target_length,
            truncation=True,
            padding="max_length"
        )
    
    # Replace padding token id with -100 (ignored by loss)
    model_inputs["labels"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]
    
    return model_inputs


def evaluate_generative_nli(
    model: T5ForConditionalGeneration,
    tokenizer: T5TokenizerFast,
    dataset: Dataset,
    max_input_length: int = 256,
    max_target_length: int = 16
) -> Dict:
    """
    Evaluate generative NLI model.
    
    Args:
        model: T5 model
        tokenizer: T5 tokenizer
        dataset: Dataset to evaluate
        max_input_length: Maximum input length
        max_target_length: Maximum target length
        
    Returns:
        Dictionary with accuracy metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    predictions = []
    references = []
    
    # Reverse label map for text to ID
    text_to_id = {v: k for k, v in T5_LABEL_MAP.items()}
    
    for example in tqdm(dataset, desc="Evaluating"):
        input_text = f"nli premise: {example['premise']} hypothesis: {example['hypothesis']}"
        
        inputs = tokenizer(
            input_text,
            max_length=max_input_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_target_length,
                num_beams=1  # Greedy decoding for NLI
            )
        
        predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        # Map predicted text to label ID
        predicted_id = text_to_id.get(predicted_text, -1)  # -1 for invalid predictions
        
        predictions.append(predicted_id)
        references.append(example["label"])
    
    predictions = np.array(predictions)
    references = np.array(references)
    
    # Calculate accuracy
    accuracy = (predictions == references).mean()
    
    # Calculate per-class accuracy
    metrics = {"accuracy": accuracy}
    for label_id, label_name in ID_TO_LABEL.items():
        mask = references == label_id
        if mask.sum() > 0:
            metrics[f"accuracy_{label_name}"] = (predictions[mask] == references[mask]).mean()
    
    return metrics


def train_generative_nli(
    model_checkpoint: str,
    dataset_name: str,
    output_dir: str,
    max_input_length: int = 256,
    max_target_length: int = 16,
    batch_size: int = 8,
    num_epochs: int = 3,
    learning_rate: float = 1e-4,
    seed: int = 42
) -> Dict:
    """
    Train generative NLI model (T5).
    
    Args:
        model_checkpoint: Path to pretrained T5 model
        dataset_name: Name of NLI dataset
        output_dir: Directory to save fine-tuned model
        max_input_length: Maximum input length
        max_target_length: Maximum target length
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        seed: Random seed
        
    Returns:
        Dictionary with evaluation metrics
    """
    set_seed(seed)
    
    logger.info("="*80)
    logger.info("Generative NLI Fine-tuning (T5)")
    logger.info(f"Model: {model_checkpoint}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Output: {output_dir}")
    logger.info("="*80)
    
    # Load tokenizer
    try:
        tokenizer = T5TokenizerFast.from_pretrained(
            model_checkpoint,
            local_files_only=True
        )
        logger.info("✓ Loaded tokenizer from checkpoint")
    except:
        logger.warning("Using fallback t5-small tokenizer")
        tokenizer = T5TokenizerFast.from_pretrained("t5-small")
    
    # Load dataset
    raw_datasets = load_nli_dataset(dataset_name)
    
    # Preprocess
    logger.info("Preprocessing datasets...")
    tokenized_datasets = raw_datasets.map(
        lambda x: preprocess_generative_nli(x, tokenizer, max_input_length, max_target_length),
        batched=True,
        remove_columns=raw_datasets["train"].column_names
    )
    
    # Load model
    model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        predict_with_generate=True,
        save_total_limit=1,
        logging_steps=100,
        gradient_accumulation_steps=2,
        fp16=False,
        report_to=[],
        seed=seed,
    )
    
    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Evaluate
    logger.info("Evaluating model...")
    metrics = evaluate_generative_nli(
        model,
        tokenizer,
        raw_datasets["validation"],
        max_input_length=max_input_length,
        max_target_length=max_target_length
    )
    
    # Save results
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Also save human-readable results
    with open(os.path.join(output_dir, "results.txt"), 'w') as f:
        f.write("NLI Generative Results (T5)\n")
        f.write("="*50 + "\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
    
    logger.info("Training completed successfully!")
    return metrics


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fine-tune models on NLI tasks")
    
    # Required arguments
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        required=True,
        help="Path to pretrained model"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=["xnli-en", "xnli-fr", "anli", "mnli"],
        help="NLI dataset name"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for fine-tuned model"
    )
    
    # Model type
    parser.add_argument(
        "--model_type",
        type=str,
        default="classification",
        choices=["classification", "generative"],
        help="Model type (classification for BERT-like, generative for T5)"
    )
    
    # Training configuration
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--max_input_length", type=int, default=256)
    parser.add_argument("--max_target_length", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Create output directory
    ensure_dir(args.output_dir)
    
    # Train based on model type
    if args.model_type == "classification":
        metrics = train_classification_nli(
            model_checkpoint=args.model_checkpoint,
            dataset_name=args.dataset_name,
            output_dir=args.output_dir,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            seed=args.seed
        )
        accuracy_key = "eval_accuracy"
    else:  # generative
        metrics = train_generative_nli(
            model_checkpoint=args.model_checkpoint,
            dataset_name=args.dataset_name,
            output_dir=args.output_dir,
            max_input_length=args.max_input_length,
            max_target_length=args.max_target_length,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            seed=args.seed
        )
        accuracy_key = "accuracy"
    
    print("\n" + "="*80)
    print("FINE-TUNING COMPLETED")
    print("="*80)
    print(f"Accuracy: {metrics.get(accuracy_key, metrics.get('accuracy', 0)):.4f}")
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()