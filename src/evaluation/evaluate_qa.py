#!/usr/bin/env python3
"""
evaluate_qa.py
Standalone evaluation script for Question Answering models

Evaluates fine-tuned QA models on validation/test sets and saves:
- Metrics (F1, Exact Match)
- Predictions
- N-best predictions
"""

import os
import json
import argparse
import collections
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import evaluate

# Disable wandb
os.environ["WANDB_DISABLED"] = "true"


class QAEvaluator:
    """Evaluator for extractive QA models (BERT-like)."""
    
    def __init__(
        self,
        model_path: str,
        validation_data_path: str,
        data_field: str = "data",
        max_length: int = 128
    ):
        self.model_path = model_path
        self.validation_data_path = validation_data_path
        self.data_field = data_field
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading model from: {model_path}")
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        print(f"Loading dataset from: {validation_data_path}")
        self.raw_datasets = load_dataset(
            "json",
            data_files={"validation": validation_data_path},
            field=self.data_field
        )
        
        print(f"Preprocessing dataset...")
        self.validation_dataset = self._preprocess_dataset()
        print(f"✓ Ready to evaluate {len(self.validation_dataset)} examples")

    def _preprocess_dataset(self) -> Dataset:
        """Preprocess the validation dataset."""
        return self.raw_datasets["validation"].map(
            self._preprocess_examples,
            batched=True,
            remove_columns=self.raw_datasets["validation"].column_names,
        )

    def _preprocess_examples(self, examples: Dict) -> Dict:
        """Preprocess individual examples."""
        questions = [q.strip() for q in examples["question"]]
        
        inputs = self.tokenizer(
            questions,
            examples["context"],
            max_length=self.max_length,
            truncation=True,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

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

    def predict(self, batch_size: int = 16) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using the model."""
        self.model.eval()
        all_start_logits = []
        all_end_logits = []

        # Prepare dataset for PyTorch
        dataset = self.validation_dataset.remove_columns(
            ['example_id', 'offset_mapping']
        ).with_format("torch")

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        for batch in tqdm(dataloader, desc="Predicting"):
            inputs = {
                key: batch[key].to(self.device)
                for key in self.tokenizer.model_input_names
            }
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            all_start_logits.append(outputs.start_logits.cpu().numpy())
            all_end_logits.append(outputs.end_logits.cpu().numpy())

        start_logits = np.concatenate(all_start_logits, axis=0)
        end_logits = np.concatenate(all_end_logits, axis=0)
        
        return start_logits, end_logits

    def compute_metrics(
        self,
        start_logits: np.ndarray,
        end_logits: np.ndarray,
        n_best: int = 20,
        max_answer_length: int = 30
    ) -> Tuple[Dict, List, List]:
        """Compute evaluation metrics."""
        metric = evaluate.load("squad")
        
        # Map examples to features
        example_to_features = collections.defaultdict(list)
        for idx, feature in enumerate(self.validation_dataset):
            example_to_features[feature["example_id"]].append(idx)

        predicted_answers = []
        nbest_predictions = []

        for example in tqdm(self.raw_datasets["validation"], desc="Computing metrics"):
            example_id = example["id"]
            context = example["context"]
            answers = []

            # Process all features for this example
            for feature_index in example_to_features[example_id]:
                start_logit = start_logits[feature_index]
                end_logit = end_logits[feature_index]
                offsets = self.validation_dataset[feature_index]["offset_mapping"]

                # Get top-k start and end positions
                start_indexes = np.argsort(start_logit)[-1: -n_best - 1: -1].tolist()
                end_indexes = np.argsort(end_logit)[-1: -n_best - 1: -1].tolist()

                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Skip invalid predictions
                        if offsets[start_index] is None or offsets[end_index] is None:
                            continue
                        if end_index < start_index:
                            continue
                        if end_index - start_index + 1 > max_answer_length:
                            continue

                        answers.append({
                            "text": context[offsets[start_index][0]: offsets[end_index][1]],
                            "logit_score": start_logit[start_index] + end_logit[end_index]
                        })

            # Select best answer
            if answers:
                best_answer = max(answers, key=lambda x: x["logit_score"])
                predicted_answers.append({
                    "id": example_id,
                    "prediction_text": best_answer["text"]
                })
                nbest_predictions.append({
                    "id": example_id,
                    "nbest": sorted(answers, key=lambda x: x["logit_score"], reverse=True)[:n_best]
                })
            else:
                predicted_answers.append({
                    "id": example_id,
                    "prediction_text": ""
                })
                nbest_predictions.append({
                    "id": example_id,
                    "nbest": []
                })

        # Prepare references
        references = [
            {"id": ex["id"], "answers": ex["answers"]}
            for ex in self.raw_datasets["validation"]
        ]
        
        # Compute metrics
        metrics = metric.compute(predictions=predicted_answers, references=references)

        return metrics, predicted_answers, nbest_predictions


class ResultSaver:
    """Handle saving evaluation results."""
    
    @staticmethod
    def save_all(
        output_dir: str,
        metrics: Dict,
        predictions: List,
        nbest_predictions: List,
        dataset_name: str
    ):
        """Save all results to files."""
        os.makedirs(output_dir, exist_ok=True)

        # Save metrics (human-readable)
        metrics_path = os.path.join(output_dir, f"{dataset_name}_results.txt")
        with open(metrics_path, "w") as f:
            f.write(f"Question Answering Evaluation Results\n")
            f.write(f"{'='*50}\n")
            f.write(f"F1 Score:      {metrics['f1']:.4f} ({metrics['f1']:.2f}%)\n")
            f.write(f"Exact Match:   {metrics['exact_match']:.4f} ({metrics['exact_match']:.2f}%)\n")

        # Save metrics (JSON)
        metrics_json_path = os.path.join(output_dir, f"{dataset_name}_metrics.json")
        with open(metrics_json_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # Save predictions
        predictions_path = os.path.join(output_dir, f"{dataset_name}_predictions.json")
        with open(predictions_path, "w") as f:
            json.dump(predictions, f, indent=2)

        # Save n-best predictions
        nbest_path = os.path.join(output_dir, f"{dataset_name}_nbest_predictions.json")
        with open(nbest_path, "w") as f:
            json.dump(
                nbest_predictions,
                f,
                indent=2,
                default=ResultSaver._convert_to_serializable
            )

        print(f"\n{'='*60}")
        print(f"Results saved to: {output_dir}")
        print(f"{'='*60}")
        print(f"  Metrics:            {metrics_path}")
        print(f"  Metrics (JSON):     {metrics_json_path}")
        print(f"  Predictions:        {predictions_path}")
        print(f"  N-best predictions: {nbest_path}")

    @staticmethod
    def _convert_to_serializable(obj):
        """Convert numpy types to Python native types."""
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj


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
        description="Evaluate QA model and save predictions"
    )
    
    # Required arguments
    parser.add_argument(
        "model_dir",
        type=str,
        help="Directory containing model (or checkpoint)"
    )
    parser.add_argument(
        "--validation_data",
        type=str,
        required=True,
        help="Path to validation data JSON file"
    )
    
    # Optional arguments
    parser.add_argument(
        "--data_field",
        type=str,
        default="data",
        help="Field in JSON containing the data (default: 'data')"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="qa",
        help="Name to use in output files (default: 'qa')"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: same as model_dir)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for evaluation (default: 16)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length (default: 128)"
    )
    
    args = parser.parse_args()

    # Set output directory
    output_dir = args.output_dir if args.output_dir else args.model_dir

    try:
        # Find checkpoint
        checkpoint_path = find_last_checkpoint(args.model_dir)
        print(f"Using checkpoint: {checkpoint_path}")

        # Initialize evaluator
        evaluator = QAEvaluator(
            checkpoint_path,
            args.validation_data,
            args.data_field,
            args.max_length
        )

        # Make predictions
        print("\nMaking predictions...")
        start_logits, end_logits = evaluator.predict(batch_size=args.batch_size)

        # Compute metrics
        print("\nComputing metrics...")
        metrics, predictions, nbest_predictions = evaluator.compute_metrics(
            start_logits,
            end_logits
        )

        # Save results
        ResultSaver.save_all(
            output_dir,
            metrics,
            predictions,
            nbest_predictions,
            args.dataset_name
        )

        # Print summary
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"F1 Score:    {metrics['f1']:.4f}")
        print(f"Exact Match: {metrics['exact_match']:.4f}")
        print(f"{'='*60}")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()