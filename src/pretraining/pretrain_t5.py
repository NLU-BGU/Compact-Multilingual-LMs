"""
train_t5.py
T5 pretraining script with span-corruption objective for multilingual experiments

T5 (Text-to-Text Transfer Transformer) uses span corruption pretraining:
- Randomly masks contiguous spans of tokens
- Model learns to predict the masked spans
- Encoder-decoder architecture suitable for sequence-to-sequence tasks

Based on: Raffel et al. (2020) "Exploring the Limits of Transfer Learning"
"""

import os
import logging
import argparse
from pathlib import Path
from typing import List, Optional

import torch
from datasets import Dataset, load_dataset
from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    T5TokenizerFast,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    set_seed,
)

from t5_mlm_collator import DataCollatorForT5MLM

# Disable wandb logging
os.environ["WANDB_DISABLED"] = "true"

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ============================================================================
# DTensor Compatibility Fix
# ============================================================================

def fix_dtensor_compatibility():
    """
    Workaround for DTensor compatibility issues in some PyTorch versions.
    This prevents errors when using certain transformers features.
    """
    import transformers.pytorch_utils
    
    if not hasattr(torch.distributed, "tensor") or \
       not hasattr(torch.distributed.tensor, "DTensor"):
        transformers.pytorch_utils.id_tensor_storage = lambda t: id(t.storage())


# ============================================================================
# Data Loading
# ============================================================================

def load_text_dataset(file_path: str, use_cache: bool = True) -> Dataset:
    """
    Load text dataset from file.
    
    Args:
        file_path: Path to text file (one sentence per line)
        use_cache: Whether to use HuggingFace datasets cache
        
    Returns:
        Dataset object with 'text' column
    """
    logger.info(f"Loading dataset from: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Read lines from file
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Loaded {len(lines):,} lines from file")
    
    # Create dataset
    dataset = Dataset.from_dict({"text": lines})
    logger.info(f"Created dataset with {len(dataset):,} examples")
    
    return dataset


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: T5TokenizerFast,
    max_length: int = 128,
    num_proc: int = 4
) -> Dataset:
    """
    Tokenize text dataset.
    
    Args:
        dataset: Input dataset with 'text' column
        tokenizer: T5 tokenizer
        max_length: Maximum sequence length
        num_proc: Number of processes for parallel tokenization
        
    Returns:
        Tokenized dataset
    """
    logger.info("Tokenizing dataset...")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=["text"],
        desc="Tokenizing"
    )
    
    logger.info(f"Tokenization complete. Dataset size: {len(tokenized_dataset):,}")
    return tokenized_dataset


# ============================================================================
# Model Setup
# ============================================================================

def setup_model_and_tokenizer(
    model_name_or_path: str,
    vocab_size: Optional[int] = None
) -> tuple:
    """
    Initialize T5 model and tokenizer.
    
    Args:
        model_name_or_path: Pretrained model name or path to config
        vocab_size: Custom vocabulary size (optional)
        
    Returns:
        Tuple of (model, tokenizer, config)
    """
    logger.info(f"Loading tokenizer from: {model_name_or_path}")
    tokenizer = T5TokenizerFast.from_pretrained(model_name_or_path)
    
    logger.info(f"Loading config from: {model_name_or_path}")
    config = T5Config.from_pretrained(model_name_or_path)
    
    # Update vocab size if needed
    if vocab_size is not None and config.vocab_size != vocab_size:
        logger.warning(f"Updating config vocab_size: {config.vocab_size} → {vocab_size}")
        config.vocab_size = vocab_size
    elif config.vocab_size != len(tokenizer):
        logger.warning(
            f"Updating config vocab_size to match tokenizer: "
            f"{config.vocab_size} → {len(tokenizer)}"
        )
        config.vocab_size = len(tokenizer)
    
    logger.info("Initializing T5 model from config...")
    model = T5ForConditionalGeneration(config)
    model.resize_token_embeddings(len(tokenizer))
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model initialized with {num_params:,} parameters")
    
    return model, tokenizer, config


# ============================================================================
# Training
# ============================================================================

def train_t5(
    data_path: str,
    output_dir: str,
    model_name_or_path: str = "t5-small",
    seed: int = 42,
    max_seq_length: int = 128,
    batch_size: int = 8,
    learning_rate: float = 5e-4,
    weight_decay: float = 0.01,
    num_epochs: int = 3,
    max_steps: Optional[int] = None,
    warmup_steps: int = 500,
    logging_steps: int = 500,
    save_steps: int = 2000,
    save_total_limit: int = 2,
    noise_density: float = 0.15,
    mean_noise_span_length: int = 3
):
    """
    Train T5 model with span-corruption objective.
    
    Args:
        data_path: Path to training data file
        output_dir: Directory to save model checkpoints
        model_name_or_path: Base model configuration
        seed: Random seed
        max_seq_length: Maximum sequence length
        batch_size: Training batch size per device
        learning_rate: Learning rate
        weight_decay: Weight decay for AdamW
        num_epochs: Number of training epochs
        max_steps: Maximum training steps (overrides num_epochs if set)
        warmup_steps: Number of warmup steps
        logging_steps: Log every N steps
        save_steps: Save checkpoint every N steps
        save_total_limit: Maximum number of checkpoints to keep
        noise_density: Proportion of tokens to mask
        mean_noise_span_length: Average length of masked spans
    """
    # Setup
    set_seed(seed)
    fix_dtensor_compatibility()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    logger.info("=" * 80)
    logger.info("T5 Training Configuration:")
    logger.info(f"  Data: {data_path}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Model: {model_name_or_path}")
    logger.info(f"  Seed: {seed}")
    logger.info(f"  Max length: {max_seq_length}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Epochs: {num_epochs}")
    if max_steps:
        logger.info(f"  Max steps: {max_steps}")
    logger.info(f"  Noise density: {noise_density}")
    logger.info(f"  Mean span length: {mean_noise_span_length}")
    logger.info("=" * 80)
    
    # Load model and tokenizer
    model, tokenizer, config = setup_model_and_tokenizer(model_name_or_path)
    model.to(device)
    
    # Load and tokenize dataset
    dataset = load_text_dataset(data_path)
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, max_seq_length)
    
    # Create data collator for span corruption
    data_collator = DataCollatorForT5MLM(
        tokenizer=tokenizer,
        noise_density=noise_density,
        mean_noise_span_length=mean_noise_span_length
    )
    logger.info(
        f"Using T5 MLM data collator "
        f"(noise_density={noise_density}, mean_span_length={mean_noise_span_length})"
    )
    
    # Setup training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        fp16=torch.cuda.is_available(),
        report_to=[],
        seed=seed,
        logging_first_step=True,
    )
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    logger.info(f"Saving final model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info("Training completed successfully!")


# ============================================================================
# Model Naming
# ============================================================================

def create_model_name(data_path: str, seed: int) -> str:
    """Create standardized model name"""
    base_name = os.path.basename(data_path).replace("+", "_").replace(".txt", "")
    return f"T5-tiny-{base_name}-seed{seed}"


def get_output_path(model_name: str, base_dir: str) -> str:
    """Get full output path for model"""
    return os.path.join(base_dir, model_name)


# ============================================================================
# Multi-Seed Training
# ============================================================================

def train_with_seeds(
    data_path: str,
    output_dir: str,
    seeds: List[int],
    **kwargs
):
    """
    Train T5 models with multiple seeds.
    
    Args:
        data_path: Path to training data
        output_dir: Base output directory
        seeds: List of random seeds
        **kwargs: Additional arguments passed to train_t5
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path {data_path} does not exist!")
    
    if len(seeds) == 0:
        raise ValueError("At least one seed must be provided.")
    
    data_name = os.path.basename(data_path)
    
    for seed in seeds:
        print("\n" + "=" * 80)
        print(f"Training T5: data={data_name}, seed={seed}")
        print("=" * 80 + "\n")
        
        model_name = create_model_name(data_path, seed)
        full_output_dir = get_output_path(model_name, output_dir)
        
        train_t5(
            data_path=data_path,
            output_dir=full_output_dir,
            seed=seed,
            **kwargs
        )


# ============================================================================
# Command-Line Interface
# ============================================================================

def main():
    """Main function for command-line execution"""
    parser = argparse.ArgumentParser(
        description="Train T5 model with span-corruption objective"
    )
    
    # Required arguments
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help="Path to training data file"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help="Output directory for model checkpoints"
    )
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        required=True,
        help="List of random seeds (e.g., --seeds 1 2 3)"
    )
    
    # Model configuration
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        default="t5-small",
        help="Pretrained model name or path (default: t5-small)"
    )
    parser.add_argument(
        '--max_seq_length',
        type=int,
        default=128,
        help="Maximum sequence length (default: 128)"
    )
    
    # Training configuration
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help="Training batch size per device (default: 8)"
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=5e-4,
        help="Learning rate (default: 5e-4)"
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.01,
        help="Weight decay (default: 0.01)"
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=None,
        help="Maximum training steps (overrides num_epochs)"
    )
    parser.add_argument(
        '--warmup_steps',
        type=int,
        default=500,
        help="Number of warmup steps (default: 500)"
    )
    parser.add_argument(
        '--logging_steps',
        type=int,
        default=500,
        help="Log every N steps (default: 500)"
    )
    parser.add_argument(
        '--save_steps',
        type=int,
        default=2000,
        help="Save checkpoint every N steps (default: 2000)"
    )
    
    # Span corruption configuration
    parser.add_argument(
        '--noise_density',
        type=float,
        default=0.15,
        help="Proportion of tokens to mask (default: 0.15)"
    )
    parser.add_argument(
        '--mean_noise_span_length',
        type=int,
        default=3,
        help="Average length of masked spans (default: 3)"
    )
    
    args = parser.parse_args()
    
    # Train with all seeds
    train_with_seeds(
        data_path=args.data_path,
        output_dir=args.output_dir,
        seeds=args.seeds,
        model_name_or_path=args.model_name_or_path,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        noise_density=args.noise_density,
        mean_noise_span_length=args.mean_noise_span_length
    )


if __name__ == "__main__":
    main()