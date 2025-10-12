"""
train_babyberta.py
BabyBERTa pretraining script for multilingual experiments

Code adapted from: https://github.com/phueb/BabyBERTa/tree/master/huggingface_recommended
"""

import os
import logging
import argparse
from pathlib import Path
from typing import List, Any, Tuple, Optional
from itertools import islice

import torch
import numpy as np
from torch.utils.data import DataLoader
from tokenizers import ByteLevelBPETokenizer
from datasets import Dataset, DatasetDict
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizerFast
)

# Disable wandb logging
os.environ["WANDB_DISABLED"] = "true"

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================================
# Configuration Constants
# ============================================================================

DEFAULT_VOCAB_SIZE = 8000
DEFAULT_MIN_FREQUENCY = 2
DEFAULT_HIDDEN_SIZE = 256
DEFAULT_NUM_LAYERS = 8
DEFAULT_NUM_HEADS = 8
DEFAULT_INTERMEDIATE_SIZE = 1024
DEFAULT_MAX_LENGTH = 128
DEFAULT_MLM_PROBABILITY = 0.15

# Training hyperparameters
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_MAX_STEPS = 260_000
DEFAULT_WARMUP_STEPS = 24_000
DEFAULT_SAVE_STEPS = 40_000
DEFAULT_SAVE_TOTAL_LIMIT = 1

# ============================================================================
# Custom Trainer
# ============================================================================

class CustomTrainer(Trainer):
    """Custom Trainer that disables shuffling in the DataLoader."""
    
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=False,  # Disable shuffling to maintain data order
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )


# ============================================================================
# Custom Data Collator
# ============================================================================

class CustomDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    """
    Custom data collator for masked language modeling without token replacement.
    Uses 90% masking, 10% random replacement strategy.
    """
    
    def torch_mask_tokens(
        self, 
        inputs: Any, 
        special_tokens_mask: Optional[Any] = None
    ) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling.
        90% MASK, 10% random, 0% original.
        """
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                ) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Only compute loss on masked tokens

        # 90% of the time, replace masked input tokens with tokenizer.mask_token
        indices_replaced = torch.bernoulli(
            torch.full(labels.shape, 0.9)
        ).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # 10% of the time, replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.1)).bool() 
            & masked_indices 
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels


# ============================================================================
# Data Loading and Processing
# ============================================================================

def load_sentences_from_file(
    file_path: Path,
    include_punctuation: bool = True,
    allow_discard: bool = False,
    min_words: int = 3
) -> List[str]:
    """
    Load sentences from a line-by-line text file.
    
    Args:
        file_path: Path to the input text file
        include_punctuation: Whether to keep sentence-final punctuation
        allow_discard: Whether to discard sentences shorter than min_words
        min_words: Minimum number of words per sentence
        
    Returns:
        List of sentence strings
    """
    print(f'Loading {file_path}', flush=True)
    sentences = []
    num_too_small = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            sentence = line.strip()
            if not sentence:
                continue

            # Discard sentences that are too short
            if sentence.count(' ') < min_words - 1 and allow_discard:
                num_too_small += 1
                continue

            # Remove sentence-final punctuation if requested
            if not include_punctuation:
                sentence = sentence.rstrip('.!?')

            sentences.append(sentence)

    if num_too_small:
        print(
            f'WARNING: Skipped {num_too_small:,} sentences '
            f'shorter than {min_words} words.'
        )

    print(f'Loaded {len(sentences):,} sentences', flush=True)
    return sentences


def make_sequences(
    sentences: List[str],
    num_sentences_per_input: int
) -> List[str]:
    """
    Combine multiple sentences into sequences.
    
    Args:
        sentences: List of individual sentences
        num_sentences_per_input: Number of sentences to combine per sequence
        
    Returns:
        List of combined sequences
    """
    gen = (s for s in sentences)
    sequences = []
    
    while True:
        sentences_in_sequence = list(islice(gen, 0, num_sentences_per_input))
        if not sentences_in_sequence:
            break
        sequence = ' '.join(sentences_in_sequence)
        sequences.append(sequence)
    
    print(f'Created {len(sequences):,} sequences', flush=True)
    return sequences


# ============================================================================
# Model and Tokenizer Setup
# ============================================================================

def train_tokenizer(
    data_path: str,
    model_name: str,
    output_dir: str,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    min_frequency: int = DEFAULT_MIN_FREQUENCY
) -> RobertaTokenizerFast:
    """
    Train a byte-level BPE tokenizer and save it.
    
    Args:
        data_path: Path to training data
        model_name: Name for the tokenizer files
        output_dir: Directory to save tokenizer files
        vocab_size: Vocabulary size
        min_frequency: Minimum token frequency
        
    Returns:
        Trained RobertaTokenizerFast
    """
    print("Training tokenizer...", flush=True)
    
    # Train byte-level BPE tokenizer
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=data_path,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<s>", "</s>", "<pad>", "<unk>", "<mask>"]
    )
    
    # Save tokenizer
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_model(output_dir, prefix=model_name)
    
    # Load as RobertaTokenizerFast
    tokenizer_fast = RobertaTokenizerFast(
        vocab_file=os.path.join(output_dir, f"{model_name}-vocab.json"),
        merges_file=os.path.join(output_dir, f"{model_name}-merges.txt"),
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        mask_token="<mask>"
    )
    
    print(f"Tokenizer trained with vocab size: {len(tokenizer_fast)}", flush=True)
    return tokenizer_fast


def initialize_model(vocab_size: int = DEFAULT_VOCAB_SIZE) -> RobertaForMaskedLM:
    """
    Initialize a RoBERTa model from scratch with BabyBERTa configuration.
    
    Args:
        vocab_size: Size of the vocabulary
        
    Returns:
        Initialized RobertaForMaskedLM model
    """
    print("Initializing RoBERTa model from scratch...", flush=True)
    
    config = RobertaConfig(
        vocab_size=vocab_size,
        hidden_size=DEFAULT_HIDDEN_SIZE,
        num_hidden_layers=DEFAULT_NUM_LAYERS,
        num_attention_heads=DEFAULT_NUM_HEADS,
        intermediate_size=DEFAULT_INTERMEDIATE_SIZE
    )
    
    model = RobertaForMaskedLM(config)
    print(f"Model initialized with {model.num_parameters():,} parameters", flush=True)
    
    return model


# ============================================================================
# Utility Functions
# ============================================================================

def create_model_name(data_path: str, seed: int, unmasking: bool) -> str:
    """
    Create a standardized model name based on configuration.
    
    Args:
        data_path: Path to training data
        seed: Random seed
        unmasking: Whether standard unmasking is used
        
    Returns:
        Formatted model name string
    """
    base_name = os.path.basename(data_path).replace("+", "_").replace(".txt", "")
    mask_status = "with" if unmasking else "without"
    model_name = f"BabyBERTa-{base_name}-{mask_status}-Masking-seed{seed}"
    return model_name


def get_pretraining_model_path(model_name: str, base_dir: str) -> str:
    """
    Get the full path for saving the pretrained model.
    
    Args:
        model_name: Name of the model
        base_dir: Base directory for saving models
        
    Returns:
        Full path for model directory
    """
    model_name = os.path.basename(os.path.normpath(model_name))
    return os.path.join(base_dir, model_name)


def setup_logger(name: str = __name__) -> logging.Logger:
    """Set up and return a configured logger."""
    logger = logging.getLogger(name)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    logger.setLevel(logging.INFO)
    return logger


# ============================================================================
# Main Training Function
# ============================================================================

def train(
    data_path: str,
    seed: int,
    unmasking: bool,
    output_base_dir: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    max_steps: int = DEFAULT_MAX_STEPS,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    save_steps: int = DEFAULT_SAVE_STEPS
):
    """
    Train a BabyBERTa model.
    
    Args:
        data_path: Path to training data file
        seed: Random seed for reproducibility
        unmasking: Whether to use standard unmasking (True) or custom (False)
        output_base_dir: Base directory for saving models and tokenizers
        batch_size: Training batch size
        learning_rate: Learning rate
        max_steps: Maximum training steps
        warmup_steps: Number of warmup steps
        save_steps: Save checkpoint every N steps
    """
    # Setup
    logger = setup_logger()
    set_seed(seed)
    
    model_name = create_model_name(data_path, seed, unmasking)
    output_dir = get_pretraining_model_path(model_name, output_base_dir)
    
    logger.info(f"=" * 80)
    logger.info(f"Training Configuration:")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Data: {data_path}")
    logger.info(f"  Seed: {seed}")
    logger.info(f"  Unmasking: {unmasking}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"=" * 80)
    
    # Training arguments
    training_args = TrainingArguments(
        report_to=None,
        output_dir=output_dir,
        overwrite_output_dir=False,
        do_train=True,
        do_eval=False,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        seed=seed,
        save_steps=save_steps,
        save_total_limit=DEFAULT_SAVE_TOTAL_LIMIT,
        logging_steps=1000,
        logging_first_step=True
    )
    
    # Load and process data
    logger.info("Loading data...")
    sentences = load_sentences_from_file(
        data_path,
        include_punctuation=True,
        allow_discard=True
    )
    sequences = make_sequences(sentences, num_sentences_per_input=1)
    
    data_dict = {'text': sequences}
    datasets = DatasetDict({'train': Dataset.from_dict(data_dict)})
    
    # Train tokenizer
    tokenizer = train_tokenizer(
        data_path=data_path,
        model_name=model_name,
        output_dir=output_dir
    )
    
    # Initialize model
    model = initialize_model(vocab_size=len(tokenizer))
    
    # Tokenize dataset
    def tokenize_function(examples):
        # Filter out empty lines
        examples["text"] = [
            line for line in examples["text"]
            if len(line) > 0 and not line.isspace()
        ]
        return tokenizer(
            examples["text"],
            padding=True,
            truncation=True,
            max_length=DEFAULT_MAX_LENGTH,
            return_special_tokens_mask=True
        )
    
    logger.info("Tokenizing dataset...")
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=["text"],
        load_from_cache_file=True
    )
    train_dataset = tokenized_datasets["train"]
    
    # Select data collator
    if unmasking:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=DEFAULT_MLM_PROBABILITY
        )
        logger.info("Using standard DataCollatorForLanguageModeling")
    else:
        data_collator = CustomDataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=DEFAULT_MLM_PROBABILITY
        )
        logger.info("Using CustomDataCollatorForLanguageModeling")
    
    # Initialize trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    logger.info(f"Saving final model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    logger.info("Training completed successfully!")


# ============================================================================
# Multi-Seed Training
# ============================================================================

def train_with_seeds(
    data_path: str,
    seeds: List[int],
    output_base_dir: str,
    unmasking_options: List[bool] = [False]
):
    """
    Train models with multiple seeds and unmasking configurations.
    
    Args:
        data_path: Path to training data
        seeds: List of random seeds
        output_base_dir: Base directory for outputs
        unmasking_options: List of unmasking configurations to try
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path {data_path} does not exist!")
    
    if len(seeds) == 0:
        raise ValueError("At least one seed must be provided.")
    
    data_name = os.path.basename(data_path)
    
    for unmasking in unmasking_options:
        for seed in seeds:
            print("\n" + "=" * 80)
            print(f"Training: data={data_name}, seed={seed}, unmasking={unmasking}")
            print("=" * 80 + "\n")
            
            train(
                data_path=data_path,
                seed=seed,
                unmasking=unmasking,
                output_base_dir=output_base_dir
            )


# ============================================================================
# Command-Line Interface
# ============================================================================

def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(
        description="Train BabyBERTa model with specified configuration."
    )
    
    # Required arguments
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help="Path to the training data file (.txt format, one sentence per line)"
    )
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        required=True,
        help="List of random seeds for training (e.g., --seeds 1 2 3)"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help="Base directory for saving models and tokenizers"
    )
    
    # Optional arguments
    parser.add_argument(
        '--unmasking',
        action='store_true',
        help="Use standard unmasking strategy (default: custom 90-10 strategy)"
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Training batch size (default: {DEFAULT_BATCH_SIZE})"
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"Learning rate (default: {DEFAULT_LEARNING_RATE})"
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"Maximum training steps (default: {DEFAULT_MAX_STEPS:,})"
    )
    
    args = parser.parse_args()
    
    # Determine unmasking options
    unmasking_options = [args.unmasking]
    
    # Train with all seeds
    train_with_seeds(
        data_path=args.data_path,
        seeds=args.seeds,
        output_base_dir=args.output_dir,
        unmasking_options=unmasking_options
    )


if __name__ == "__main__":
    main()