#!/usr/bin/env python3
"""
evaluate_nli.py
Standalone evaluation script for Natural Language Inference models

Evaluates fine-tuned NLI models on validation/test sets and saves:
- Metrics (Accuracy, F1, per-class metrics)
- Predictions
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
import evaluate

# Disable wandb
os.environ["WANDB_DISABLED"] = "true"

# Label mappings
ID_TO_LABEL = {0: "entailment", 1: "neutral", 2: "contradiction"}


class NLIEvaluator:
    """Evaluator for NLI classification models."""
    
    def __init__(
        self,
        model_path: str,
        validation_file: str,
        max_seq_length: int = 128
    ):
        self.model_path = model_path
        self.validation_file = validation_file
        self.max_seq_length = max_seq_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading model from: {model_path}")
        self.model = self._load_model().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        print(f"Loading dataset from: {validation_file}")
        self.dataset = self._load_and_preprocess_dataset()
        print(f"✓ Ready to evaluate {len(self.dataset)} examples")
    
    def _load_model(self) -> AutoModelForSequenceClassification:
        """Load model with fallback for different weight formats."""
        SAFE_WEIGHTS_NAME = "model.safetensors"
        WEIGHTS_NAME = "pytorch_model.bin"
        
        if os.path.exists(os.path.join(self.model_path, SAFE_WEIGHTS_NAME)):
            return AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                use_safetensors=True,
                trust_remote_code=True
            )
        elif os.path.exists(os.path.join(self.model_path, WEIGHTS_NAME)):
            return AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
        else:
            # Try default loading
            return AutoModelForSequenceClassification.from_pretrained(
                self.model_path
            )
    
    def _load_and_preprocess_dataset(self) -> Dataset:
        """Load and tokenize the dataset."""
        dataset = load_dataset(
            "json",
            data_files={"validation": self.validation_file}
        )["validation"]
        
        def tokenize_function(example):
            return self.tokenizer(
                example["premise"],
                example["hypothesis"],
                truncation=True,
                padding="max_length",
                max_length=self.max_seq_length
            )
        
        dataset = dataset.map(tokenize_function, batched=True)
        dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "label"]
        )
        
        return dataset
    
    def predict(self, batch_size: int = 32) -> tuple[List[int], List[int]]:
        """Make predictions on the dataset."""
        self.model.eval()
        predictions = []
        labels = []
        
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size
        )
        
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = {
                k: v.to(self.device)
                for k, v in batch.items()
                if k in self.tokenizer.model_input_names
            }
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            logits = outputs.logits.cpu().numpy()
            batch_preds = np.argmax(logits, axis=1)
            
            predictions.extend(batch_preds)
            labels.extend(batch["label"].cpu().numpy())
        
        return predictions, labels
    
    def compute_metrics(
        self,
        predictions: List[int],
        labels: List[int]
    ) -> Dict:
        """Compute evaluation metrics."""
        # Load metrics
        accuracy_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1")
        
        # Compute overall metrics
        accuracy = accuracy_metric.compute(
            predictions=predictions,
            references=labels
        )["accuracy"]
        
        f1_macro = f1_metric.compute(
            predictions=predictions,
            references=labels,
            average="macro"
        )["f1"]
        
        f1_micro = f1_metric.compute(
            predictions=predictions,
            references=labels,
            average="micro"
        )["f1"]
        
        # Compute per-class metrics
        per_class_metrics = {}
        for label_id, label_name in ID_TO_LABEL.items():
            mask = np.array(labels) == label_id
            if mask.sum() > 0:
                class_accuracy = (
                    np.array(predictions)[mask] == np.array(labels)[mask]
                ).mean()
                per_class_metrics[f"accuracy_{label_name}"] = float(class_accuracy)
        
        # Combine all metrics
        metrics = {
            "accuracy": float(accuracy),
            "f1_macro": float(f1_macro),
            "f1_micro": float(f1_micro),
            **per_class_metrics
        }
        
        return metrics


class ResultSaver:
    """Handle saving evaluation results."""
    
    @staticmethod
    def save_all(
        output_dir: str,
        metrics: Dict,
        predictions: List[int],
        labels: List[int],
        dataset_name: str
    ):
        """Save all results to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics (human-readable)
        metrics_path = os.path.join(output_dir, f"{dataset_name}_results.txt")
        with open(metrics_path, "w") as f:
            f.write("Natural Language Inference Evaluation Results\n")
            f.write("="*60 + "\n\n")
            f.write(f"Overall Metrics:\n")
            f.write(f"  Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
            f.write(f"  F1 (macro):  {metrics['f1_macro']:.4f}\n")
            f.write(f"  F1 (micro):  {metrics['f1_micro']:.4f}\n\n")
            
            f.write(f"Per-Class Accuracy:\n")
            for label_id, label_name in ID_TO_LABEL.items():
                key = f"accuracy_{label_name}"
                if key in metrics:
                    f.write(f"  {label_name.capitalize():<15}: {metrics[key]:.4f} ({metrics[key]*100:.2f}%)\n")
        
        # Save metrics (JSON)
        metrics_json_path = os.path.join(output_dir, f"{dataset_name}_metrics.json")
        with open(metrics_json_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Save predictions
        predictions_data = [
            {
                "prediction": int(pred),
                "label": int(label),
                "prediction_text": ID_TO_LABEL[pred],
                "label_text": ID_TO_LABEL[label],
                "correct": int(pred) == int(label)
            }
            for pred, label in zip(predictions, labels)
        ]
        
        predictions_path = os.path.join(output_dir, f"{dataset_name}_predictions.json")
        with open(predictions_path, "w") as f:
            json.dump(predictions_data, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Results saved to: {output_dir}")
        print(f"{'='*60}")
        print(f"  Metrics:        {metrics_path}")
        print(f"  Metrics (JSON): {metrics_json_path}")
        print(f"  Predictions:    {predictions_path}")


def find_last_checkpoint(path: str) -> str:
    """Find the last checkpoint in the given path."""
    if not os.path.exists(path):
        raise ValueError(f"Path does not exist: {path}")
    
    # If path is already a checkpoint, use it
    if "checkpoint" in os.path.basename(path):
        return path
    
    # Look for checkpoint subdirectories
    subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    checkpoint_dirs = [d for d in subdirs if d.startswith("checkpoint-")]
    
    if not checkpoint_dirs:
        # No checkpoints found, assume path is the model
        return path
    
    # Sort by checkpoint number and return the latest
    checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]), reverse=True)
    return os.path.join(path, checkpoint_dirs[0])


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate NLI model and save predictions"
    )
    
    # Required arguments
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the model checkpoint"
    )
    parser.add_argument(
        "validation_file",
        type=str,
        help="Path to the XNLI validation JSON file"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save evaluation results (default: same as model_path)"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="nli",
        help="Name to use in output files (default: 'nli')"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation (default: 32)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length (default: 128)"
    )
    
    args = parser.parse_args()
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else args.model_path
    
    try:
        # Find checkpoint
        checkpoint_path = find_last_checkpoint(args.model_path)
        print(f"Using checkpoint: {checkpoint_path}")
        
        # Initialize evaluator
        evaluator = NLIEvaluator(
            checkpoint_path,
            args.validation_file,
            args.max_length
        )
        
        # Make predictions
        print("\nMaking predictions...")
        predictions, labels = evaluator.predict(batch_size=args.batch_size)
        
        # Compute metrics
        print("\nComputing metrics...")
        metrics = evaluator.compute_metrics(predictions, labels)
        
        # Save results
        ResultSaver.save_all(
            output_dir,
            metrics,
            predictions,
            labels,
            args.dataset_name
        )
        
        # Print summary
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Accuracy:   {metrics['accuracy']:.4f}")
        print(f"F1 (macro): {metrics['f1_macro']:.4f}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()