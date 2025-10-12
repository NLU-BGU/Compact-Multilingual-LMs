#!/usr/bin/env python3
"""
evaluate_grammar.py
Grammar evaluation script for masked language models (BabyBERTa, RoBERTa, etc.)

Evaluates models on grammar acceptability tasks like BLiMP/CLAMS by comparing
perplexity scores between grammatical and ungrammatical sentences.

The model should assign lower perplexity (higher probability) to
grammatically correct sentences.
"""

import os
import argparse
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import numpy as np
from transformers import (
    RobertaTokenizer,
    RobertaForMaskedLM,
    RobertaConfig,
    RobertaTokenizerFast
)
from tokenizers import ByteLevelBPETokenizer

# Disable wandb
os.environ["WANDB_DISABLED"] = "true"


def get_perplexity(model, tokenizer, sentence: str) -> float:
    """
    Calculate perplexity for a sentence using masked language modeling.
    
    Args:
        model: Masked language model
        tokenizer: Tokenizer
        sentence: Input sentence
        
    Returns:
        Perplexity score (lower is better)
    """
    try:
        tensor_input = tokenizer.encode(sentence, return_tensors='pt')
        
        # Check if tokens are within vocabulary range
        vocab_size = model.config.vocab_size
        max_token = tensor_input.max().item()
        
        if max_token >= vocab_size:
            print(f"Warning: Token {max_token} exceeds vocab size {vocab_size}")
            # Replace out-of-vocab tokens with UNK
            tensor_input = torch.clamp(tensor_input, max=vocab_size - 1)
        
        # Ensure we have enough tokens
        if tensor_input.size(-1) < 3:
            return 1000.0  # Return high perplexity for very short sentences
        
        # Create masked versions of the sentence
        repeat_input = tensor_input.repeat(tensor_input.size(-1) - 2, 1)
        mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
        masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id)
        
        # Ensure mask token is within vocab
        if tokenizer.mask_token_id >= vocab_size:
            safe_mask_id = tokenizer.unk_token_id if hasattr(tokenizer, 'unk_token_id') else 3
            masked_input = repeat_input.masked_fill(mask == 1, safe_mask_id)
        
        labels = repeat_input.masked_fill(masked_input != tokenizer.mask_token_id, -100)
        
        # Move to device
        device = next(model.parameters()).device
        masked_input = masked_input.to(device)
        labels = labels.to(device)
        
        # Calculate loss and convert to perplexity
        with torch.inference_mode():
            loss = model(masked_input, labels=labels).loss
        
        return np.exp(loss.item())
        
    except Exception as e:
        print(f"Error calculating perplexity for '{sentence}': {e}")
        return 1000.0


def evaluate_grammar_tsv(
    model,
    tokenizer,
    file_path: str,
    verbose: bool = False
) -> Tuple[float, int, int]:
    """
    Evaluate grammar using TSV format (label\tsentence).
    
    Args:
        model: Masked language model
        tokenizer: Tokenizer
        file_path: Path to TSV file
        verbose: Print detailed results
        
    Returns:
        Tuple of (accuracy, correct_predictions, total_pairs)
    """
    sentences = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) != 2:
                continue
            
            label, sentence = parts
            sentences.append((label.strip() == 'True', sentence.strip()))
    
    # Group sentences into pairs (True, False)
    correct_predictions = 0
    total_pairs = 0
    
    for i in range(0, len(sentences), 2):
        if i + 1 >= len(sentences):
            break
        
        true_sent = sentences[i]
        false_sent = sentences[i + 1]
        
        # Verify we have a True/False pair
        if not (true_sent[0] == True and false_sent[0] == False):
            if verbose:
                print(f"Warning: Unexpected pair at lines {i+1}-{i+2}")
            continue
        
        # Calculate perplexity for both sentences
        true_perplexity = get_perplexity(model, tokenizer, true_sent[1])
        false_perplexity = get_perplexity(model, tokenizer, false_sent[1])
        
        # Model should assign lower perplexity to correct sentence
        if true_perplexity < false_perplexity:
            correct_predictions += 1
        
        total_pairs += 1
        
        if verbose:
            print(f"  Grammatical:   '{true_sent[1]}' (PPL: {true_perplexity:.2f})")
            print(f"  Ungrammatical: '{false_sent[1]}' (PPL: {false_perplexity:.2f})")
            print(f"  Correct: {'✓' if true_perplexity < false_perplexity else '✗'}")
            print()
    
    accuracy = correct_predictions / total_pairs if total_pairs > 0 else 0
    return accuracy, correct_predictions, total_pairs


def create_tokenizer_from_data(
    data_path: str,
    output_dir: str = "./temp_tokenizer"
) -> RobertaTokenizerFast:
    """
    Create tokenizer from training data (matches pretraining tokenizer).
    
    Args:
        data_path: Path to training data file
        output_dir: Directory to save temporary tokenizer files
        
    Returns:
        RobertaTokenizerFast
    """
    print(f"Building tokenizer from data: {data_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Build tokenizer exactly like in training
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=data_path,
        vocab_size=8000,
        min_frequency=2,
        special_tokens=["<s>", "</s>", "<pad>", "<unk>", "<mask>"]
    )
    
    # Save tokenizer files
    tokenizer.save_model(output_dir, prefix="temp_tokenizer")
    
    # Load as RobertaTokenizerFast
    fast_tokenizer = RobertaTokenizerFast(
        vocab_file=f"{output_dir}/temp_tokenizer-vocab.json",
        merges_file=f"{output_dir}/temp_tokenizer-merges.txt",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        mask_token="<mask>"
    )
    
    print(f"✓ Tokenizer built with vocab size: {fast_tokenizer.vocab_size}")
    return fast_tokenizer


def load_model_only(model_path: str) -> RobertaForMaskedLM:
    """
    Load model without tokenizer.
    
    Args:
        model_path: Path to model checkpoint
        
    Returns:
        Loaded model
    """
    model_dir = Path(model_path)
    
    # Load config
    config_path = model_dir / "config.json"
    if config_path.exists():
        config = RobertaConfig.from_json_file(str(config_path))
    else:
        # Default BabyBERTa config
        config = RobertaConfig(
            vocab_size=8000,
            hidden_size=256,
            num_hidden_layers=8,
            num_attention_heads=8,
            intermediate_size=1024,
            initializer_range=0.02,
        )
    
    # Create model
    model = RobertaForMaskedLM(config)
    
    # Load weights
    from safetensors.torch import load_file
    
    safetensors_path = model_dir / "model.safetensors"
    if safetensors_path.exists():
        state_dict = load_file(str(safetensors_path))
        model.load_state_dict(state_dict, strict=False)
    else:
        raise FileNotFoundError(f"model.safetensors not found in {model_path}")
    
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    return model


def evaluate_single_model(
    model_path: str,
    tokenizer,
    data_dir: str
) -> Dict[str, float]:
    """
    Evaluate a single model on all grammar test files.
    
    Args:
        model_path: Path to model checkpoint
        tokenizer: Tokenizer to use
        data_dir: Directory containing test files
        
    Returns:
        Dictionary mapping test names to accuracy scores
    """
    model = load_model_only(model_path)
    model_name = Path(model_path).parent.name
    
    print(f"  Evaluating {model_name}...")
    
    # Find all test files
    txt_files = glob.glob(f"{data_dir}/*.txt")
    results = {}
    
    for file_path in txt_files:
        file_name = Path(file_path).stem
        accuracy, correct, total = evaluate_grammar_tsv(
            model,
            tokenizer,
            file_path,
            verbose=False
        )
        results[file_name] = accuracy
    
    # Calculate average
    if results:
        avg_accuracy = np.mean(list(results.values()))
        print(f"    Average: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
    
    return results


def evaluate_model_group(
    base_model_path: str,
    training_data_path: str,
    seeds: List[int],
    data_dir: str
) -> Tuple[Dict, any]:
    """
    Evaluate a group of models with different seeds.
    
    Args:
        base_model_path: Base model path (with seed42 as placeholder)
        training_data_path: Path to training data for tokenizer
        seeds: List of seeds to evaluate
        data_dir: Directory containing test files
        
    Returns:
        Tuple of (all_results, tokenizer)
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING MODEL GROUP: {Path(base_model_path).name}")
    print(f"Training data: {Path(training_data_path).name}")
    print(f"Seeds: {seeds}")
    print('='*80)
    
    # Build tokenizer once for this group
    print("Building tokenizer...")
    tokenizer = create_tokenizer_from_data(training_data_path)
    
    # Evaluate each seed
    all_results = {}
    for seed in seeds:
        model_path = base_model_path.replace("seed42", f"seed{seed}")
        model_path = f"{model_path}/checkpoint-260000"
        
        if Path(model_path).exists():
            results = evaluate_single_model(model_path, tokenizer, data_dir)
            all_results[f"seed{seed}"] = results
        else:
            print(f"  Warning: Model path not found: {model_path}")
    
    return all_results, tokenizer


def save_results(
    base_model_name: str,
    all_results: Dict,
    training_data_path: str,
    output_dir: str = "results"
):
    """
    Save evaluation results to file.
    
    Args:
        base_model_name: Base model name
        all_results: Results dictionary
        training_data_path: Path to training data
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Extract base name without seed
    model_base = base_model_name.replace("-seed42", "")
    
    # Save results
    output_file = f"{output_dir}/{model_base}_grammar_results_{timestamp}.txt"
    
    with open(output_file, 'w') as f:
        f.write(f"Grammar Evaluation Results\n")
        f.write(f"{'='*60}\n")
        f.write(f"Model Group: {model_base}\n")
        f.write(f"Training Data: {training_data_path}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if not all_results:
            f.write("No results to report.\n")
            return output_file
        
        # Get all test names
        test_names = set()
        for seed_results in all_results.values():
            test_names.update(seed_results.keys())
        test_names = sorted(test_names)
        
        # Write header
        f.write(f"{'Test':<30}")
        for seed in sorted(all_results.keys()):
            f.write(f"{seed:>10}")
        f.write(f"{'Mean':>10}{'Std':>8}\n")
        f.write("-" * (30 + len(all_results) * 10 + 18) + "\n")
        
        # Write results for each test
        for test_name in test_names:
            f.write(f"{test_name:<30}")
            scores = []
            
            for seed in sorted(all_results.keys()):
                score = all_results[seed].get(test_name, 0.0)
                f.write(f"{score:>10.4f}")
                scores.append(score)
            
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            f.write(f"{mean_score:>10.4f}{std_score:>8.4f}\n")
        
        # Overall statistics
        f.write("-" * (30 + len(all_results) * 10 + 18) + "\n")
        
        overall_means = []
        for seed_results in all_results.values():
            if seed_results:
                overall_means.append(np.mean(list(seed_results.values())))
        
        if overall_means:
            f.write(f"{'OVERALL':<30}")
            for mean in overall_means:
                f.write(f"{mean:>10.4f}")
            f.write(f"{np.mean(overall_means):>10.4f}{np.std(overall_means):>8.4f}\n")
    
    print(f"\n✓ Results saved to: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate grammar acceptability for masked language models"
    )
    
    parser.add_argument(
        "--base_model_path",
        required=True,
        help="Base model path (use 'seed42' as placeholder, e.g., 'models/babyberta-seed42')"
    )
    parser.add_argument(
        "--training_data_path",
        required=True,
        help="Path to training data for building tokenizer"
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 51, 71],
        help="Seeds to evaluate (default: 42 51 71)"
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Directory containing grammar test files (.txt format)"
    )
    parser.add_argument(
        "--output_dir",
        default="results",
        help="Output directory for results (default: results)"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.training_data_path).exists():
        print(f"Error: Training data not found: {args.training_data_path}")
        return 1
    
    if not Path(args.data_dir).exists():
        print(f"Error: Test directory not found: {args.data_dir}")
        return 1
    
    # Extract base model name
    base_model_name = Path(args.base_model_path).name
    
    print(f"Starting grammar evaluation...")
    print(f"Base model: {base_model_name}")
    print(f"Training data: {args.training_data_path}")
    print(f"Seeds: {args.seeds}")
    print(f"Grammar tests: {args.data_dir}")
    
    # Run evaluation
    all_results, tokenizer = evaluate_model_group(
        args.base_model_path,
        args.training_data_path,
        args.seeds,
        args.data_dir
    )
    
    # Save results
    output_file = save_results(
        base_model_name,
        all_results,
        args.training_data_path,
        args.output_dir
    )
    
    # Print summary
    if all_results:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print('='*60)
        
        for seed, results in all_results.items():
            if results:
                avg = np.mean(list(results.values()))
                print(f"{seed}: {avg:.4f} ({avg*100:.2f}%)")
        
        # Overall average across seeds
        all_avgs = [
            np.mean(list(results.values()))
            for results in all_results.values()
            if results
        ]
        if all_avgs:
            overall_avg = np.mean(all_avgs)
            overall_std = np.std(all_avgs)
            print(f"\nOverall: {overall_avg:.4f} ± {overall_std:.4f}")
    
    return 0


if __name__ == "__main__":
    exit(main())