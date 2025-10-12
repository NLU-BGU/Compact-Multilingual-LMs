"""
t5_mlm_collator.py
Data collator for T5 span corruption pretraining

Implements span masking strategy where contiguous spans of tokens are replaced
with sentinel tokens, and the model learns to predict the masked spans.

Example:
    Input:  "Thank you for inviting me to your party last week"
    Masked: "Thank you <X> me to your party <Y> week"
    Target: "<X> for inviting <Y> last <Z>"
"""

import torch
import random
from typing import List, Dict, Any


class DataCollatorForT5MLM:
    """
    Data collator for T5-style span corruption masked language modeling.
    
    This collator implements the span corruption objective used in T5 pretraining:
    1. Randomly samples multiple spans from the input
    2. Replaces each span with a unique sentinel token (e.g., <extra_id_0>)
    3. Creates target sequence with sentinels followed by the original span tokens
    
    Args:
        tokenizer: T5 tokenizer with sentinel tokens
        noise_density: Proportion of tokens to mask (default: 0.15)
        mean_noise_span_length: Average length of each masked span (default: 3)
        input_length: Maximum input sequence length (default: 128)
    
    Example:
        >>> collator = DataCollatorForT5MLM(tokenizer, noise_density=0.15, mean_noise_span_length=3)
        >>> batch = collator(examples)
        >>> # batch contains 'input_ids' (corrupted) and 'labels' (targets)
    """
    
    def __init__(
        self,
        tokenizer,
        noise_density: float = 0.15,
        mean_noise_span_length: int = 3,
        input_length: int = 128
    ):
        self.tokenizer = tokenizer
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.input_length = input_length
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        
        # Validate sentinel tokens exist
        self._validate_sentinel_tokens()
    
    def _validate_sentinel_tokens(self):
        """Ensure tokenizer has sentinel tokens for span masking"""
        try:
            sentinel_test = self.tokenizer.convert_tokens_to_ids("<extra_id_0>")
            if sentinel_test is None or sentinel_test == self.tokenizer.unk_token_id:
                raise ValueError(
                    "Tokenizer does not have sentinel tokens (<extra_id_N>). "
                    "Please use a T5 tokenizer with sentinel tokens."
                )
        except Exception as e:
            raise ValueError(f"Error validating sentinel tokens: {e}")
    
    def _create_span_mask(self, num_tokens: int) -> List[int]:
        """
        Create list of span starting positions for masking.
        
        Args:
            num_tokens: Total number of tokens in sequence
            
        Returns:
            Sorted list of span start positions
        """
        num_mask = max(1, int(num_tokens * self.noise_density))
        # Sample random starting positions
        span_starts = sorted(random.sample(range(num_tokens), min(num_mask, num_tokens)))
        return span_starts
    
    def _corrupt_sequence(
        self,
        input_ids: List[int],
        span_starts: List[int]
    ) -> tuple[List[int], List[int]]:
        """
        Apply span corruption to input sequence.
        
        Args:
            input_ids: Original token IDs
            span_starts: Starting positions of spans to mask
            
        Returns:
            Tuple of (corrupted_ids, label_ids)
        """
        num_tokens = len(input_ids)
        corrupted = []
        labels = []
        current = 0
        sentinel = 0
        
        for start in span_starts:
            # Skip if this span overlaps with previous
            if start < current:
                continue
            
            # Calculate span end position
            end = min(start + self.mean_noise_span_length, num_tokens)
            
            # Add tokens before span (unchanged)
            if current < start:
                corrupted.extend(input_ids[current:start])
            
            # Add sentinel token to corrupted sequence
            sentinel_id = self.tokenizer.convert_tokens_to_ids(f"<extra_id_{sentinel}>")
            corrupted.append(sentinel_id)
            
            # Add sentinel and masked tokens to labels
            labels.append(sentinel_id)
            labels.extend(input_ids[start:end])
            
            sentinel += 1
            current = end
        
        # Add remaining tokens after last span
        if current < num_tokens:
            corrupted.extend(input_ids[current:])
        
        return corrupted, labels
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Process batch of examples with span corruption.
        
        Args:
            examples: List of examples, each with 'input_ids' key
            
        Returns:
            Dictionary with 'input_ids' (corrupted) and 'labels' (targets) tensors
        """
        batch_input_ids = []
        batch_labels = []
        
        for ex in examples:
            input_ids = ex["input_ids"]
            
            # Truncate to leave space for EOS token
            input_ids = input_ids[:self.input_length - 1]
            num_tokens = len(input_ids)
            
            # Create span mask
            span_starts = self._create_span_mask(num_tokens)
            
            # Apply span corruption
            corrupted, labels = self._corrupt_sequence(input_ids, span_starts)
            
            # Add EOS token to both sequences
            corrupted.append(self.eos_token_id)
            labels.append(self.eos_token_id)
            
            # Truncate to max length
            corrupted = corrupted[:self.input_length]
            labels = labels[:self.input_length]
            
            # Pad to max length
            pad_len_input = self.input_length - len(corrupted)
            pad_len_labels = self.input_length - len(labels)
            
            corrupted += [self.pad_token_id] * pad_len_input
            labels += [self.pad_token_id] * pad_len_labels
            
            batch_input_ids.append(corrupted)
            batch_labels.append(labels)
        
        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }