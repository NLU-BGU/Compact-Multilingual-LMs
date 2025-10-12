"""
finetune_qa.py
Fine-tuning script for Question Answering tasks

Supports multiple model architectures:
- BabyBERTa / RoBERTa (extractive QA)
- T5 (generative QA)
- LTG-BERT (extractive QA)

Supports multiple QA datasets:
- SQuAD (English)
- French SQuAD
- QAMR (English)
- QA-SRL (English)
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
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
# Dataset Loading and Preprocessing
# ============================================================================

def flatten_squad_format(dataset: Dataset) -> Dataset:
    """
    Flatten nested SQuAD-format dataset structure.
    
    Converts from nested paragraphs/qas structure to flat structure
    with one example per question.
    
    Args:
        dataset: Dataset in SQuAD format with nested structure
        
    Returns:
        Flattened dataset
    """
    flat_data = []
    
    for article in dataset:
        title = article.get('title', '')
        
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            
            for qa in paragraph['qas']:
                flat_entry = {
                    'id': qa['id'],
                    'title': title,
                    'context': context,
                    'question': qa['question'],
                    'answers': {
                        'answer_start': [a['answer_start'] for a in qa['answers']],
                        'text': [a['text'] for a in qa['answers']]
                    }
                }
                flat_data.append(flat_entry)
    
    return Dataset.from_pandas(pd.DataFrame(flat_data))


def load_qa_dataset(
    dataset_name: str,
    base_data_dir: str = "/sise/eliorsu-group/lielbin/Research/datasets/data-finetune"
) -> DatasetDict:
    """
    Load question answering dataset by name.
    
    Args:
        dataset_name: Name of dataset (squad, fr-squad, qamr, qasrl)
        base_data_dir: Base directory containing datasets
        
    Returns:
        DatasetDict with train and validation splits
    """
    logger.info(f"Loading dataset: {dataset_name}")
    
    paths = get_dataset_paths(dataset_name, base_data_dir)
    
    # Load dataset
    if dataset_name == "fr-squad":
        # French SQuAD needs flattening
        datasets = load_dataset(
            'json',
            data_files={'train': paths['train'], 'validation': paths['validation']},
            field='data'
        )
        raw_datasets = DatasetDict({
            'train': flatten_squad_format(datasets['train']),
            'validation': flatten_squad_format(datasets['validation'])
        })
    else:
        # Other datasets are pre-flattened
        raw_datasets = load_dataset(
            'json',
            data_files={'train': paths['train'], 'validation': paths['validation']},
            field='data'
        )
    
    logger.info(f"Loaded {len(raw_datasets['train'])} train examples")
    logger.info(f"Loaded {len(raw_datasets['validation'])} validation examples")
    
    return raw_datasets


# ============================================================================
# Extractive QA (BabyBERTa, RoBERTa, BERT)
# ============================================================================

def preprocess_extractive_qa(
    examples: Dict,
    tokenizer: AutoTokenizer,
    max_length: int = 128,
    is_training: bool = True
) -> Dict:
    """
    Preprocess examples for extractive QA models.
    
    Args:
        examples: Batch of examples
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        is_training: Whether preprocessing for training
        
    Returns:
        Preprocessed examples with tokenized inputs
    """
    questions = [q.strip() for q in examples["question"]]
    
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation=True,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    
    if is_training:
        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []
        
        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            
            if not answer["text"]:
                start_positions.append(0)
                end_positions.append(0)
                continue
            
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)
            
            # Find context boundaries
            idx = 0
            while idx < len(sequence_ids) and sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            
            while idx < len(sequence_ids) and sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1
            
            # Check if answer is outside context
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Find start position
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)
                
                # Find end position
                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)
        
        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
    else:
        # Validation preprocessing
        sample_map = inputs.pop("overflow_to_sample_mapping")
        example_ids = []
        
        for i in range(len(inputs["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(examples["id"][sample_idx])
            
            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None
                for k, o in enumerate(offset)
            ]
        
        inputs["example_id"] = example_ids
    
    return inputs


def compute_extractive_qa_metrics(
    start_logits: np.ndarray,
    end_logits: np.ndarray,
    features: Dataset,
    examples: Dataset,
    n_best: int = 20,
    max_answer_length: int = 30
) -> Dict:
    """
    Compute metrics for extractive QA.
    
    Args:
        start_logits: Predicted start positions
        end_logits: Predicted end positions
        features: Tokenized features
        examples: Original examples
        n_best: Number of best answers to consider
        max_answer_length: Maximum answer length
        
    Returns:
        Dictionary with F1 and Exact Match scores
    """
    import collections
    
    # Map examples to features
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)
    
    predicted_answers = []
    
    for example in tqdm(examples, desc="Computing predictions"):
        example_id = example["id"]
        context = example["context"]
        answers = []
        
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]
            
            start_indexes = np.argsort(start_logit)[-1: -n_best - 1: -1].tolist()
            end_indexes = np.argsort(end_logit)[-1: -n_best - 1: -1].tolist()
            
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    if end_index < start_index:
                        continue
                    if end_index - start_index + 1 > max_answer_length:
                        continue
                    
                    answer = {
                        "text": context[offsets[start_index][0]: offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index]
                    }
                    answers.append(answer)
        
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append({
                "id": example_id,
                "prediction_text": best_answer["text"]
            })
        else:
            predicted_answers.append({
                "id": example_id,
                "prediction_text": ""
            })
    
    # Prepare references
    references = [
        {"id": ex["id"], "answers": ex["answers"]}
        for ex in examples
    ]
    
    # Compute SQuAD metrics
    metric = evaluate.load("squad")
    return metric.compute(predictions=predicted_answers, references=references)


def train_extractive_qa(
    model_checkpoint: str,
    dataset_name: str,
    output_dir: str,
    max_seq_length: int = 128,
    batch_size: int = 8,
    num_epochs: int = 3,
    learning_rate: float = 1e-4,
    seed: int = 42
):
    """
    Train extractive QA model (BabyBERTa, RoBERTa, BERT).
    
    Args:
        model_checkpoint: Path to pretrained model
        dataset_name: Name of QA dataset
        output_dir: Directory to save fine-tuned model
        max_seq_length: Maximum sequence length
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        seed: Random seed
    """
    set_seed(seed)
    
    logger.info("="*80)
    logger.info("Extractive QA Fine-tuning")
    logger.info(f"Model: {model_checkpoint}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Output: {output_dir}")
    logger.info("="*80)
    
    # Load tokenizer and dataset
    tokenizer = load_tokenizer(model_checkpoint)
    raw_datasets = load_qa_dataset(dataset_name)
    
    # Preprocess datasets
    logger.info("Preprocessing datasets...")
    train_dataset = raw_datasets["train"].map(
        lambda x: preprocess_extractive_qa(x, tokenizer, max_seq_length, is_training=True),
        batched=True,
        remove_columns=raw_datasets["train"].column_names
    )
    
    validation_dataset = raw_datasets["validation"].map(
        lambda x: preprocess_extractive_qa(x, tokenizer, max_seq_length, is_training=False),
        batched=True,
        remove_columns=raw_datasets["validation"].column_names
    )
    
    # Load model
    from transformers import AutoModelForQuestionAnswering
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=2,
        save_total_limit=1,
        logging_steps=100,
        report_to=[],
        seed=seed,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
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
    predictions, _, _ = trainer.predict(validation_dataset)
    start_logits, end_logits = predictions
    
    metrics = compute_extractive_qa_metrics(
        start_logits,
        end_logits,
        validation_dataset,
        raw_datasets["validation"]
    )
    
    # Save results
    logger.info(f"F1: {metrics['f1']:.4f}, Exact Match: {metrics['exact_match']:.4f}")
    
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info("Training completed successfully!")
    return metrics


# ============================================================================
# Generative QA (T5)
# ============================================================================

def preprocess_generative_qa(
    examples: Dict,
    tokenizer: T5TokenizerFast,
    max_input_length: int = 512,
    max_target_length: int = 128
) -> Dict:
    """
    Preprocess examples for generative QA (T5).
    
    Args:
        examples: Batch of examples
        tokenizer: T5 tokenizer
        max_input_length: Maximum input length
        max_target_length: Maximum target length
        
    Returns:
        Preprocessed examples
    """
    # Create input in T5 format
    inputs = [
        f"question: {q} context: {c}"
        for q, c in zip(examples["question"], examples["context"])
    ]
    
    # Extract target answers
    targets = [
        answers["text"][0] if answers["text"] else ""
        for answers in examples["answers"]
    ]
    
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


def evaluate_generative_qa(
    model: T5ForConditionalGeneration,
    tokenizer: T5TokenizerFast,
    dataset: DatasetDict,
    split: str = "validation",
    max_input_length: int = 512,
    max_target_length: int = 128,
    max_samples: Optional[int] = None
) -> Dict:
    """
    Evaluate generative QA model.
    
    Args:
        model: T5 model
        tokenizer: T5 tokenizer
        dataset: Dataset to evaluate
        split: Dataset split to use
        max_input_length: Maximum input length
        max_target_length: Maximum target length
        max_samples: Maximum number of samples to evaluate
        
    Returns:
        Dictionary with F1 and Exact Match scores
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    eval_dataset = dataset[split]
    if max_samples:
        eval_dataset = eval_dataset.select(range(min(max_samples, len(eval_dataset))))
    
    predicted_answers = []
    references = []
    
    for example in tqdm(eval_dataset, desc="Evaluating"):
        input_text = f"question: {example['question']} context: {example['context']}"
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
                num_beams=4
            )
        
        predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted_answers.append({
            "id": example["id"],
            "prediction_text": predicted_text
        })
        references.append({
            "id": example["id"],
            "answers": example["answers"]
        })
    
    metric = evaluate.load("squad")
    return metric.compute(predictions=predicted_answers, references=references)


def train_generative_qa(
    model_checkpoint: str,
    dataset_name: str,
    output_dir: str,
    max_input_length: int = 512,
    max_target_length: int = 128,
    batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 1e-4,
    seed: int = 42
):
    """
    Train generative QA model (T5).
    
    Args:
        model_checkpoint: Path to pretrained T5 model
        dataset_name: Name of QA dataset
        output_dir: Directory to save fine-tuned model
        max_input_length: Maximum input length
        max_target_length: Maximum target length
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        seed: Random seed
    """
    set_seed(seed)
    
    logger.info("="*80)
    logger.info("Generative QA Fine-tuning (T5)")
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
    raw_datasets = load_qa_dataset(dataset_name)
    
    # Preprocess
    logger.info("Preprocessing datasets...")
    train_dataset = raw_datasets["train"].map(
        lambda x: preprocess_generative_qa(x, tokenizer, max_input_length, max_target_length),
        batched=True,
        remove_columns=raw_datasets["train"].column_names
    )
    
    validation_dataset = raw_datasets["validation"].map(
        lambda x: preprocess_generative_qa(x, tokenizer, max_input_length, max_target_length),
        batched=True,
        remove_columns=raw_datasets["validation"].column_names
    )
    
    # Load model
    model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
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
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
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
    metrics = evaluate_generative_qa(
        model,
        tokenizer,
        raw_datasets,
        split="validation",
        max_input_length=max_input_length,
        max_target_length=max_target_length
    )
    
    # Save results
    logger.info(f"F1: {metrics['f1']:.4f}, Exact Match: {metrics['exact_match']:.4f}")
    
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info("Training completed successfully!")
    return metrics


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fine-tune models on QA tasks")
    
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
        choices=["squad", "fr-squad", "qamr", "qasrl"],
        help="QA dataset name"
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
        default="extractive",
        choices=["extractive", "generative"],
        help="Model type (extractive for BERT-like, generative for T5)"
    )
    
    # Training configuration
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Create output directory
    ensure_dir(args.output_dir)
    
    # Train based on model type
    if args.model_type == "extractive":
        metrics = train_extractive_qa(
            model_checkpoint=args.model_checkpoint,
            dataset_name=args.dataset_name,
            output_dir=args.output_dir,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            seed=args.seed
        )
    else:  # generative
        metrics = train_generative_qa(
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
    
    print("\n" + "="*80)
    print("FINE-TUNING COMPLETED")
    print("="*80)
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Exact Match: {metrics['exact_match']:.4f}")
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()