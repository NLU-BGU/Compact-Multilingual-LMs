"""
finetune_utils.py
Common utilities for fine-tuning pretrained models

Provides shared functions for:
- Tokenizer loading and fixing
- Dataset loading and preprocessing
- Model initialization
- Evaluation metrics
"""

import os
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# ============================================================================
# Tokenizer Utilities
# ============================================================================

def fix_tokenizer(model_checkpoint: str) -> AutoTokenizer:
    """
    Fix tokenizer.json format issues that can occur with custom tokenizers.
    
    Some tokenizers store merge rules as nested arrays instead of strings,
    which causes loading errors. This function converts them to the correct format.
    
    Args:
        model_checkpoint: Path to model directory containing tokenizer.json
        
    Returns:
        Loaded AutoTokenizer
        
    Raises:
        FileNotFoundError: If tokenizer.json not found
    """
    tokenizer_path = os.path.join(model_checkpoint, "tokenizer.json")
    
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"tokenizer.json not found at {tokenizer_path}")
    
    # Backup original file (only once)
    backup_path = os.path.join(model_checkpoint, "tokenizer.json.backup")
    if not os.path.exists(backup_path):
        shutil.copy2(tokenizer_path, backup_path)
        logger.info(f"Created backup: {backup_path}")
    
    # Load tokenizer config
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Check if merge rules need fixing
    if 'model' in config and 'merges' in config['model']:
        merges = config['model']['merges']
        
        # If merges are nested arrays, convert to strings
        if merges and isinstance(merges[0], list):
            logger.info("Converting merge rules from arrays to strings...")
            config['model']['merges'] = [
                f"{pair[0]} {pair[1]}" for pair in merges
            ]
            
            # Save fixed version
            with open(tokenizer_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            
            logger.info("✓ Tokenizer fixed and saved")
    
    # Load tokenizer
    return AutoTokenizer.from_pretrained(model_checkpoint)


def load_tokenizer(model_checkpoint: str, use_fast: bool = True) -> AutoTokenizer:
    """
    Load tokenizer with automatic fallback to fix_tokenizer if needed.
    
    Args:
        model_checkpoint: Path to model directory
        use_fast: Whether to use fast tokenizer implementation
        
    Returns:
        Loaded tokenizer
    """
    try:
        # Try standard loading
        tokenizer = AutoTokenizer.from_pretrained(
            model_checkpoint,
            use_fast=use_fast
        )
        logger.info(f"✓ Tokenizer loaded from {model_checkpoint}")
        return tokenizer
        
    except Exception as e:
        logger.warning(f"Standard tokenizer loading failed: {e}")
        logger.info("Attempting to fix tokenizer...")
        
        try:
            # Fallback to fix_tokenizer
            tokenizer = fix_tokenizer(model_checkpoint)
            logger.info("✓ Tokenizer loaded after fixing")
            return tokenizer
            
        except Exception as fix_error:
            logger.error(f"Failed to fix tokenizer: {fix_error}")
            raise


# ============================================================================
# Model Utilities
# ============================================================================

def load_model_for_sequence_classification(
    model_checkpoint: str,
    num_labels: int,
    device: Optional[str] = None
) -> AutoModelForSequenceClassification:
    """
    Load model for sequence classification (e.g., NLI).
    
    Args:
        model_checkpoint: Path to pretrained model
        num_labels: Number of classification labels
        device: Device to load model on (auto-detect if None)
        
    Returns:
        Model for sequence classification
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=num_labels
    )
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model.to(device)
    logger.info(f"✓ Model loaded on {device}")
    
    return model


def load_model_for_question_answering(
    model_checkpoint: str,
    device: Optional[str] = None
) -> AutoModelForQuestionAnswering:
    """
    Load model for question answering.
    
    Args:
        model_checkpoint: Path to pretrained model
        device: Device to load model on (auto-detect if None)
        
    Returns:
        Model for question answering
    """
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model.to(device)
    logger.info(f"✓ Model loaded on {device}")
    
    return model


# ============================================================================
# Path Utilities
# ============================================================================

def get_model_output_path(
    base_output_dir: str,
    model_name: str,
    task_name: str,
    seed: Optional[int] = None
) -> str:
    """
    Generate standardized output path for fine-tuned models.
    
    Args:
        base_output_dir: Base directory for saving models
        model_name: Name of the pretrained model
        task_name: Name of the fine-tuning task (e.g., 'squad', 'xnli')
        seed: Random seed (optional)
        
    Returns:
        Full output path for the fine-tuned model
    """
    # Extract model basename
    model_basename = Path(model_name).name
    
    # Create output name
    if seed is not None:
        output_name = f"{model_basename}-finetuned-{task_name}-seed{seed}"
    else:
        output_name = f"{model_basename}-finetuned-{task_name}"
    
    return os.path.join(base_output_dir, output_name)


def ensure_dir(path: str) -> str:
    """
    Ensure directory exists, create if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        The input path
    """
    os.makedirs(path, exist_ok=True)
    return path


# ============================================================================
# Dataset Path Utilities
# ============================================================================

def get_dataset_paths(dataset_name: str, base_data_dir: str) -> Dict[str, str]:
    """
    Get train/validation/test paths for a dataset.
    
    Note: This function provides default paths. You can override by passing
    custom paths directly to the training functions.
    
    Args:
        dataset_name: Name of dataset (e.g., 'squad', 'xnli', 'qamr')
        base_data_dir: Base directory containing all datasets
        
    Returns:
        Dictionary with 'train', 'validation', and optionally 'test' paths
        
    Raises:
        ValueError: If dataset_name is not recognized
    """
    paths = {}
    
    # QA Datasets
    if dataset_name == "squad":
        base = os.path.join(base_data_dir, "SQuAD", "EN")
        paths['train'] = os.path.join(base, "squad_train.json")
        paths['validation'] = os.path.join(base, "squad_dev.json")
    
    elif dataset_name == "fr-squad":
        base = os.path.join(base_data_dir, "SQuAD", "FR")
        paths['train'] = os.path.join(base, "squad_train.json")
        paths['validation'] = os.path.join(base, "squad_dev.json")
    
    elif dataset_name == "qamr-en":
        base = os.path.join(base_data_dir, "QAMR", "EN")
        paths['train'] = os.path.join(base, "qamr_train.json")
        paths['validation'] = os.path.join(base, "qamr_dev.json")
        paths['test'] = os.path.join(base, "qamr_test.json")
    
    elif dataset_name == "qamr-fr":
        base = os.path.join(base_data_dir, "QAMR", "FR")
        paths['train'] = os.path.join(base, "qamr_train.json")
        paths['validation'] = os.path.join(base, "qamr_dev.json")
        paths['test'] = os.path.join(base, "qamr_test.json")
    
    elif dataset_name == "qasrl-en":
        base = os.path.join(base_data_dir, "QASRL", "EN")
        paths['train'] = os.path.join(base, "qasrl_train.json")
        paths['validation'] = os.path.join(base, "qasrl_dev.json")
        paths['test'] = os.path.join(base, "qasrl_test.json")
    
    elif dataset_name == "qasrl-fr":
        base = os.path.join(base_data_dir, "QASRL", "FR")
        paths['train'] = os.path.join(base, "qasrl_train.json")
        paths['validation'] = os.path.join(base, "qasrl_dev.json")
        paths['test'] = os.path.join(base, "qasrl_test.json")
    
    # NLI Datasets
    elif dataset_name == "xnli-en":
        base = os.path.join(base_data_dir, "XNLI", "EN")
        paths['train'] = os.path.join(base, "xnli_train.json")
        paths['validation'] = os.path.join(base, "xnli_dev.json")
        paths['test'] = os.path.join(base, "xnli_test.json")
    
    elif dataset_name == "xnli-fr":
        base = os.path.join(base_data_dir, "XNLI", "FR")
        paths['train'] = os.path.join(base, "xnli_train.json")
        paths['validation'] = os.path.join(base, "xnli_dev.json")
        paths['test'] = os.path.join(base, "xnli_test.json")
    
    elif dataset_name == "anli":
        base = os.path.join(base_data_dir, "ANLI", "EN")
        paths['train'] = os.path.join(base, "anli_train.json")
        paths['validation'] = os.path.join(base, "anli_dev.json")
        paths['test'] = os.path.join(base, "anli_test.json")
    
    elif dataset_name == "mnli":
        base = os.path.join(base_data_dir, "MultiNLI", "EN")
        paths['train'] = os.path.join(base, "mnli_train.json")
        paths['validation'] = os.path.join(base, "mnli_dev.json")
    
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Supported: squad, fr-squad, qamr-en, qamr-fr, qasrl-en, qasrl-fr, "
            f"xnli-en, xnli-fr, anli, mnli"
        )
    
    # Validate paths exist (warning only, not error)
    for split, path in paths.items():
        if not os.path.exists(path):
            logger.warning(f"{split} file not found: {path}")
    
    return paths


# ============================================================================
# Training Configuration
# ============================================================================

def get_default_training_args(
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    **kwargs
) -> Dict[str, Any]:
    """
    Get default training arguments for fine-tuning.
    
    Args:
        output_dir: Directory to save model checkpoints
        num_epochs: Number of training epochs
        batch_size: Training batch size per device
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        **kwargs: Additional arguments to override defaults
        
    Returns:
        Dictionary of training arguments
    """
    args = {
        'output_dir': output_dir,
        'num_train_epochs': num_epochs,
        'per_device_train_batch_size': batch_size,
        'per_device_eval_batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'logging_steps': 100,
        'save_strategy': 'epoch',
        'evaluation_strategy': 'epoch',
        'save_total_limit': 2,
        'fp16': torch.cuda.is_available(),
        'report_to': [],  # Disable wandb
    }
    
    # Override with any provided kwargs
    args.update(kwargs)
    
    return args


# ============================================================================
# Seed Management
# ============================================================================

def set_seed(seed: int):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    logger.info(f"Random seed set to {seed}")


# ============================================================================
# Model Name Utilities
# ============================================================================

def extract_seed_from_path(model_path: str) -> Optional[int]:
    """
    Extract seed number from model path.
    
    Args:
        model_path: Path containing seed information (e.g., 'model-seed42')
        
    Returns:
        Seed number or None if not found
    """
    import re
    match = re.search(r'seed(\d+)', model_path)
    return int(match.group(1)) if match else None


def get_model_base_name(model_path: str) -> str:
    """
    Get base model name without seed or checkpoint suffix.
    
    Args:
        model_path: Full model path
        
    Returns:
        Base model name
    """
    name = Path(model_path).name
    
    # Remove common suffixes
    name = name.replace('-seed42', '').replace('-seed51', '').replace('-seed71', '')
    name = name.replace('checkpoint-', '')
    
    return name