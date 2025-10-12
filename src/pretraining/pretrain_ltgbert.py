"""
train_ltgbert.py
LTG-BERT pretraining script for multilingual experiments

LTG-BERT is the winning model from the BabyLM Challenge, featuring:
- NormFormer normalization for improved gradient flow
- GEGLU activation functions for enhanced expressiveness
- Disentangled attention with relative position encoding
- Span masking for effective pretraining

Based on: Samuel et al. (2023) "Trained on 100 million words and still in shape"
"""

import os
import math
import logging
import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import _softmax_backward_data
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Disable wandb logging
os.environ["WANDB_DISABLED"] = "true"

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# Configuration
# ============================================================================

class LTGBertConfig:
    """Configuration class for LTG-BERT model"""
    
    def __init__(
        self,
        vocab_size: int = 30000,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 512,
        position_bucket_size: int = 32,
        layer_norm_eps: float = 1e-12,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.position_bucket_size = position_bucket_size
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob

    @classmethod
    def from_pretrained(cls, config_path: str):
        """Load configuration from file"""
        # Implementation for loading config from JSON/YAML
        pass

    def save(self, output_path: str):
        """Save configuration to file"""
        # Implementation for saving config to JSON/YAML
        pass


# ============================================================================
# Custom Operations
# ============================================================================

class MaskedSoftmax(torch.autograd.Function):
    """Custom masked softmax with efficient gradient computation"""
    
    @staticmethod
    def forward(ctx, x, mask, dim):
        ctx.dim = dim
        x.masked_fill_(mask, float('-inf'))
        x = torch.softmax(x, ctx.dim)
        x.masked_fill_(mask, 0.0)
        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        inputGrad = _softmax_backward_data(grad_output, output, ctx.dim, output.dtype)
        return inputGrad, None, None


class GeGLU(nn.Module):
    """
    Gated Linear Unit with GELU activation (GeGLU).
    Splits input in half, applies GELU to one half, and multiplies with the other.
    """
    
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate, approximate='tanh')


# ============================================================================
# Model Components
# ============================================================================

class FeedForward(nn.Module):
    """
    Feed-forward network with GeGLU activation and NormFormer architecture.
    Uses pre-normalization and post-normalization without affine parameters.
    """
    
    def __init__(self, config: LTGBertConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False),
            nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False),
            GeGLU(),
            nn.LayerNorm(config.intermediate_size, eps=config.layer_norm_eps, elementwise_affine=False),
            nn.Linear(config.intermediate_size, config.hidden_size, bias=False),
            nn.Dropout(config.hidden_dropout_prob)
        )
        self.initialize(config.hidden_size)

    def initialize(self, hidden_size: int):
        """Initialize weights with truncated normal distribution"""
        std = math.sqrt(2.0 / (5.0 * hidden_size))
        nn.init.trunc_normal_(self.mlp[1].weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.mlp[-2].weight, mean=0.0, std=std, a=-2*std, b=2*std)

    def forward(self, x):
        return self.mlp(x)


class DisentangledAttention(nn.Module):
    """
    Multi-head attention with disentangled content and position representations.
    Implements relative position encoding with logarithmic bucketing.
    """
    
    def __init__(self, config: LTGBertConfig):
        super().__init__()

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"Hidden size {config.hidden_size} must be divisible by "
                f"number of attention heads {config.num_attention_heads}"
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_size = config.hidden_size // config.num_attention_heads

        # Content projections
        self.in_proj_qk = nn.Linear(config.hidden_size, 2 * config.hidden_size, bias=True)
        self.in_proj_v = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

        # Layer normalization
        self.pre_layer_norm = nn.LayerNorm(
            config.hidden_size, config.layer_norm_eps, elementwise_affine=False
        )
        self.post_layer_norm = nn.LayerNorm(
            config.hidden_size, config.layer_norm_eps, elementwise_affine=True
        )

        # Position encoding setup
        position_indices = self._create_position_indices(config)
        self.register_buffer("position_indices", position_indices, persistent=True)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.scale = 1.0 / math.sqrt(3 * self.head_size)
        self.initialize()

    def _create_position_indices(self, config: LTGBertConfig) -> torch.Tensor:
        """Create position indices with logarithmic bucketing"""
        position_indices = (
            torch.arange(config.max_position_embeddings, dtype=torch.long).unsqueeze(1)
            - torch.arange(config.max_position_embeddings, dtype=torch.long).unsqueeze(0)
        )
        position_indices = self._make_log_bucket_position(
            position_indices,
            config.position_bucket_size,
            config.max_position_embeddings
        )
        return config.position_bucket_size - 1 + position_indices

    def _make_log_bucket_position(
        self,
        relative_pos: torch.Tensor,
        bucket_size: int,
        max_position: int
    ) -> torch.Tensor:
        """
        Create logarithmic position buckets for efficient long-range modeling.
        Positions close to 0 are bucketed linearly, distant positions logarithmically.
        """
        sign = torch.sign(relative_pos)
        mid = bucket_size // 2
        abs_pos = torch.where(
            (relative_pos < mid) & (relative_pos > -mid),
            mid - 1,
            torch.abs(relative_pos)
        )
        log_pos = (
            torch.ceil(
                torch.log(abs_pos / mid) / math.log((max_position - 1) / mid) * (mid - 1)
            ).int() + mid
        )
        bucket_pos = torch.where(abs_pos <= mid, relative_pos, log_pos * sign).long()
        return bucket_pos

    def initialize(self):
        """Initialize projection weights"""
        std = math.sqrt(2.0 / (5.0 * self.hidden_size))
        nn.init.trunc_normal_(self.in_proj_qk.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.in_proj_v.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.out_proj.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        self.in_proj_qk.bias.data.zero_()
        self.in_proj_v.bias.data.zero_()
        self.out_proj.bias.data.zero_()

    def compute_attention_scores(
        self,
        hidden_states: torch.Tensor,
        relative_embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention scores with content and position contributions.
        
        Returns:
            attention_scores: Attention scores with shape (batch, heads, query_len, key_len)
            value: Value vectors for computing output
        """
        key_len, batch_size, _ = hidden_states.size()
        query_len = key_len

        # Apply pre-normalization
        hidden_states = self.pre_layer_norm(hidden_states)

        # Project to query, key, value
        query, key = self.in_proj_qk(hidden_states).chunk(2, dim=2)
        value = self.in_proj_v(hidden_states)

        # Get position embeddings
        pos = self.in_proj_qk(self.dropout(relative_embedding))
        pos = F.embedding(self.position_indices[:query_len, :key_len], pos)
        pos = pos.view(query_len, key_len, self.num_heads, 2 * self.head_size)
        query_pos, key_pos = pos.chunk(2, dim=3)

        # Reshape for multi-head attention
        query = query.reshape(query_len, batch_size * self.num_heads, self.head_size).transpose(0, 1)
        key = key.reshape(key_len, batch_size * self.num_heads, self.head_size).transpose(0, 1)
        value = value.view(key_len, batch_size * self.num_heads, self.head_size).transpose(0, 1)

        # Compute content-content attention scores
        attention_scores = torch.bmm(query, key.transpose(1, 2) * self.scale)

        # Add position contributions (content-position and position-content)
        query = query.view(batch_size, self.num_heads, query_len, self.head_size)
        key = key.view(batch_size, self.num_heads, key_len, self.head_size)
        attention_scores = attention_scores.view(batch_size, self.num_heads, query_len, key_len)
        
        # Content-to-position attention
        attention_scores.add_(torch.einsum("bhqd,qkhd->bhqk", query, key_pos * self.scale))
        # Position-to-content attention
        attention_scores.add_(torch.einsum("bhkd,qkhd->bhqk", key * self.scale, query_pos))

        return attention_scores, value

    def compute_output(
        self,
        attention_probs: torch.Tensor,
        value: torch.Tensor
    ) -> torch.Tensor:
        """Compute final output from attention probabilities and values"""
        attention_probs = self.dropout(attention_probs)
        context = torch.bmm(attention_probs.flatten(0, 1), value)
        context = context.transpose(0, 1).reshape(context.size(1), -1, self.hidden_size)
        context = self.out_proj(context)
        context = self.post_layer_norm(context)
        context = self.dropout(context)
        return context

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        relative_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with masked attention"""
        attention_scores, value = self.compute_attention_scores(hidden_states, relative_embedding)
        attention_probs = MaskedSoftmax.apply(attention_scores, attention_mask, -1)
        return self.compute_output(attention_probs, value)


class EncoderLayer(nn.Module):
    """Single transformer encoder layer with attention and feed-forward"""
    
    def __init__(self, config: LTGBertConfig):
        super().__init__()
        self.attention = DisentangledAttention(config)
        self.mlp = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        relative_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with residual connections"""
        x = x + self.attention(x, padding_mask, relative_embedding)
        x = x + self.mlp(x)
        return x


class Encoder(nn.Module):
    """Multi-layer transformer encoder with depth-dependent scaling"""
    
    def __init__(self, config: LTGBertConfig, activation_checkpointing: bool = False):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(config) for _ in range(config.num_hidden_layers)
        ])

        # Initialize with layer-dependent scaling for stability
        for i, layer in enumerate(self.layers):
            scale_factor = math.sqrt(1.0 / (2.0 * (1 + i)))
            layer.mlp.mlp[1].weight.data *= scale_factor
            layer.mlp.mlp[-2].weight.data *= scale_factor

        self.activation_checkpointing = activation_checkpointing
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        relative_embedding: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Forward pass through all layers.
        
        Returns:
            List of hidden states from each layer (including input)
        """
        all_hidden_states = [hidden_states]
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, relative_embedding)
            all_hidden_states.append(hidden_states)
        return all_hidden_states


class Embedding(nn.Module):
    """
    Embedding layer with word embeddings and relative position embeddings.
    Uses NormFormer-style normalization without affine parameters.
    """
    
    def __init__(self, config: LTGBertConfig):
        super().__init__()
        self.hidden_size = config.hidden_size

        # Word embeddings
        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.word_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Relative position embeddings
        self.relative_embedding = nn.Parameter(
            torch.empty(2 * config.position_bucket_size - 1, config.hidden_size)
        )
        self.relative_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.initialize()

    def initialize(self):
        """Initialize embeddings with truncated normal distribution"""
        std = math.sqrt(2.0 / (5.0 * self.hidden_size))
        nn.init.trunc_normal_(self.relative_embedding, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.word_embedding.weight, mean=0.0, std=std, a=-2*std, b=2*std)

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            word_embedding: Contextualized word embeddings
            relative_embeddings: Position embeddings for attention
        """
        word_embedding = self.dropout(
            self.word_layer_norm(self.word_embedding(input_ids))
        )
        relative_embeddings = self.relative_layer_norm(self.relative_embedding)
        return word_embedding, relative_embeddings


class MaskClassifier(nn.Module):
    """Classifier head for masked language modeling"""
    
    def __init__(self, config: LTGBertConfig, subword_embedding: nn.Parameter):
        super().__init__()
        self.nonlinearity = nn.Sequential(
            nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=False),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=False),
            nn.Linear(subword_embedding.size(1), subword_embedding.size(0))
        )
        self.initialize(config.hidden_size, subword_embedding)

    def initialize(self, hidden_size: int, embedding: nn.Parameter):
        """Initialize classifier weights, tying with input embeddings"""
        std = math.sqrt(2.0 / (5.0 * hidden_size))
        nn.init.trunc_normal_(self.nonlinearity[1].weight, mean=0.0, std=std, a=-2*std, b=2*std)
        self.nonlinearity[-1].weight = embedding  # Weight tying
        self.nonlinearity[1].bias.data.zero_()
        self.nonlinearity[-1].bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        masked_lm_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass, optionally selecting only masked positions.
        
        Args:
            x: Hidden states
            masked_lm_labels: Labels with -100 for non-masked positions
        """
        if masked_lm_labels is not None:
            # Only compute for masked positions
            mask_indices = torch.nonzero(masked_lm_labels.flatten() != -100).squeeze()
            x = torch.index_select(x.flatten(0, 1), 0, mask_indices)
        return self.nonlinearity(x)


# ============================================================================
# Main Model
# ============================================================================

class LTGBert(nn.Module):
    """
    LTG-BERT: Language Technology Group BERT
    
    Winner of the BabyLM Challenge with architectural innovations:
    - NormFormer normalization
    - GeGLU activations
    - Disentangled attention
    - Span masking pretraining
    """
    
    def __init__(
        self,
        config: LTGBertConfig,
        activation_checkpointing: bool = False
    ):
        super().__init__()
        self.config = config
        self.embedding = Embedding(config)
        self.transformer = Encoder(config, activation_checkpointing)
        self.classifier = MaskClassifier(config, self.embedding.word_embedding.weight)

    def get_contextualized(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> List[torch.Tensor]:
        """Get contextualized embeddings from all layers"""
        static_embeddings, relative_embedding = self.embedding(input_ids)
        contextualized_embeddings = self.transformer(
            static_embeddings,
            attention_mask.unsqueeze(1).unsqueeze(2),
            relative_embedding
        )
        return contextualized_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        masked_lm_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for masked language modeling.
        
        Args:
            input_ids: Input token IDs (seq_len, batch_size)
            attention_mask: Attention mask (batch_size, seq_len)
            masked_lm_labels: Labels for MLM (-100 for non-masked positions)
            
        Returns:
            Predictions for masked positions
        """
        contextualized_embeddings = self.get_contextualized(input_ids, attention_mask)[-1]
        predictions = self.classifier(contextualized_embeddings, masked_lm_labels)
        return predictions

    def save_pretrained(self, output_dir: str):
        """Save model and configuration"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model weights
        model_path = os.path.join(output_dir, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        
        # Save config
        config_path = os.path.join(output_dir, "config.json")
        self.config.save(config_path)
        
        print(f"Model saved to {output_dir}")

    @classmethod
    def from_pretrained(cls, model_dir: str):
        """Load pretrained model"""
        config_path = os.path.join(model_dir, "config.json")
        config = LTGBertConfig.from_pretrained(config_path)
        
        model = cls(config)
        
        model_path = os.path.join(model_dir, "pytorch_model.bin")
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        
        print(f"Model loaded from {model_dir}")
        return model


# ============================================================================
# Masking Strategy
# ============================================================================

class SpanMaskingStrategy:
    """
    Span masking strategy used in LTG-BERT.
    Masks contiguous spans of tokens for more challenging pretraining.
    """
    
    def __init__(
        self,
        mask_prob: float = 0.15,
        mask_token_id: int = 103,
        vocab_size: int = 30000,
        random_prob: float = 0.1,
        keep_prob: float = 0.1,
        mean_span_length: float = 3.0
    ):
        self.mask_prob = mask_prob
        self.random_prob = random_prob
        self.keep_prob = keep_prob
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.mean_span_length = mean_span_length

    def __call__(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply span masking to tokens.
        
        Args:
            tokens: Input tokens (seq_len, batch_size)
            
        Returns:
            masked_tokens: Tokens with masking applied
            labels: Labels for MLM loss (-100 for non-masked)
        """
        labels = tokens.clone()
        
        # Create span mask
        seq_len, batch_size = tokens.shape
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        
        for b in range(batch_size):
            num_to_mask = int(seq_len * self.mask_prob)
            masked_so_far = 0
            
            while masked_so_far < num_to_mask:
                # Sample span length from geometric distribution
                span_length = np.random.geometric(1 / self.mean_span_length)
                span_length = min(span_length, num_to_mask - masked_so_far)
                
                # Sample random starting position
                start = np.random.randint(0, seq_len - span_length + 1)
                mask[start:start + span_length, b] = True
                masked_so_far += span_length
        
        # Don't mask special tokens (assuming they are < 100)
        special_tokens_mask = tokens < 100
        mask &= ~special_tokens_mask
        
        # Set labels
        labels[~mask] = -100
        
        # Apply masking strategy (80% mask, 10% random, 10% keep)
        mask_indices = mask.clone()
        
        # 80% of the time: replace with [MASK]
        indices_replaced = torch.bernoulli(
            torch.full(tokens.shape, 0.8)
        ).bool() & mask_indices
        tokens[indices_replaced] = self.mask_token_id
        
        # 10% of the time: replace with random token
        indices_random = torch.bernoulli(
            torch.full(tokens.shape, 0.5)  # 0.5 of remaining 20%
        ).bool() & mask_indices & ~indices_replaced
        random_words = torch.randint(
            100, self.vocab_size, size=indices_random.sum().item(), dtype=torch.long
        )
        tokens[indices_random] = random_words
        
        # 10% of the time: keep unchanged
        
        return tokens, labels


# ============================================================================
# Data Loading
# ============================================================================

class TextDataset(Dataset):
    """Dataset for loading text data for LTG-BERT pretraining"""
    
    def __init__(
        self,
        file_path: str,
        tokenizer,
        max_length: int = 128,
        mask_prob: float = 0.15
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.masker = SpanMaskingStrategy(mask_prob=mask_prob)
        self.examples = self.load_and_tokenize(file_path)
        
    def load_and_tokenize(self, file_path: str) -> List[torch.Tensor]:
        """Load text file and tokenize"""
        print(f"Loading and tokenizing {file_path}...")
        examples = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                # Tokenize (you'll need to implement proper tokenization)
                tokens = self.tokenizer.encode(line, max_length=self.max_length)
                if len(tokens) > 10:  # Skip very short sequences
                    examples.append(torch.tensor(tokens, dtype=torch.long))
        
        print(f"Loaded {len(examples)} examples")
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.examples[idx].clone()
        
        # Create attention mask (0 for real tokens, 1 for padding)
        attention_mask = (tokens == 0)  # Assuming 0 is PAD token
        
        # Apply masking
        masked_tokens, labels = self.masker(tokens.unsqueeze(1))
        masked_tokens = masked_tokens.squeeze(1)
        labels = labels.squeeze(1)
        
        return {
            'input_ids': masked_tokens,
            'attention_mask': attention_mask,
            'labels': labels
        }


# ============================================================================
# Training
# ============================================================================

def train(
    model: LTGBert,
    dataset: TextDataset,
    output_dir: str,
    num_epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    warmup_steps: int = 1000,
    max_steps: Optional[int] = None,
    save_steps: int = 5000
):
    """
    Train LTG-BERT model.
    
    Args:
        model: LTG-BERT model to train
        dataset: Training dataset
        output_dir: Directory to save checkpoints
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        warmup_steps: Number of warmup steps
        max_steps: Maximum training steps (overrides num_epochs if set)
        save_steps: Save checkpoint every N steps
    """
    from torch.utils.data import DataLoader
    import torch.optim as optim
    
    # Setup
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    # Setup optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.98)
    )
    
    # Learning rate scheduler with warmup
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.0, (max_steps - step) / (max_steps - warmup_steps))
    
    if max_steps is None:
        max_steps = len(dataloader) * num_epochs
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    model.train()
    global_step = 0
    total_loss = 0
    
    print(f"Starting training...")
    print(f"  Total examples = {len(dataset)}")
    print(f"  Batch size = {batch_size}")
    print(f"  Total steps = {max_steps}")
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            # Move to device
            input_ids = batch['input_ids'].transpose(0, 1).to(device)  # (seq_len, batch)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].transpose(0, 1).to(device)  # (seq_len, batch)
            
            # Forward pass
            predictions = model(input_ids, attention_mask, labels)
            
            # Compute loss
            labels_flat = labels.flatten()
            mask = labels_flat != -100
            
            if mask.sum() == 0:
                continue  # Skip if no masked tokens
            
            loss = F.cross_entropy(predictions, labels_flat[mask])
            
            # Backward pass
            optimizer.zero_grad()

            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            epoch_loss += loss.item()
            total_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            # Logging
            if global_step % 100 == 0:
                avg_loss = total_loss / 100
                print(
                    f"Step {global_step}/{max_steps} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )
                total_loss = 0
            
            # Save checkpoint
            if global_step % save_steps == 0:
                checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                model.save_pretrained(checkpoint_dir)
                print(f"Checkpoint saved at step {global_step}")
            
            # Stop if max_steps reached
            if max_steps and global_step >= max_steps:
                break
        
        if max_steps and global_step >= max_steps:
            break
        
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch + 1}/{num_epochs} completed | Average Loss: {avg_epoch_loss:.4f}")
    
    # Save final model
    final_dir = os.path.join(output_dir, "final")
    model.save_pretrained(final_dir)
    print(f"Training completed! Final model saved to {final_dir}")


# ============================================================================
# Main Training Script
# ============================================================================

def create_model_name(data_path: str, seed: int) -> str:
    """Create standardized model name"""
    base_name = os.path.basename(data_path).replace("+", "_").replace(".txt", "")
    return f"LTG-BERT-{base_name}-seed{seed}"


def setup_logger(name: str = __name__) -> logging.Logger:
    """Set up logging"""
    logger = logging.getLogger(name)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    logger.setLevel(logging.INFO)
    return logger


def train_ltgbert(
    data_path: str,
    output_dir: str,
    seed: int = 42,
    vocab_size: int = 30000,
    hidden_size: int = 768,
    num_layers: int = 12,
    num_heads: int = 12,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    max_steps: int = 100000,
    warmup_steps: int = 10000,
    save_steps: int = 5000
):
    """
    Main function to train LTG-BERT model.
    
    Args:
        data_path: Path to training data
        output_dir: Output directory for model checkpoints
        seed: Random seed
        vocab_size: Vocabulary size
        hidden_size: Hidden dimension size
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        batch_size: Training batch size
        learning_rate: Learning rate
        max_steps: Maximum training steps
        warmup_steps: Warmup steps for learning rate
        save_steps: Save checkpoint every N steps
    """
    logger = setup_logger()
    
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    model_name = create_model_name(data_path, seed)
    full_output_dir = os.path.join(output_dir, model_name)
    
    logger.info("=" * 80)
    logger.info("LTG-BERT Training Configuration:")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Data: {data_path}")
    logger.info(f"  Output: {full_output_dir}")
    logger.info(f"  Vocab size: {vocab_size}")
    logger.info(f"  Hidden size: {hidden_size}")
    logger.info(f"  Layers: {num_layers}")
    logger.info(f"  Attention heads: {num_heads}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Max steps: {max_steps}")
    logger.info("=" * 80)
    
    # Create configuration
    config = LTGBertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        intermediate_size=hidden_size * 4
    )
    
    # Initialize model
    logger.info("Initializing model...")
    model = LTGBert(config)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model initialized with {num_params:,} parameters")
    
    # Load dataset (you'll need to implement proper tokenization)
    logger.info("Loading dataset...")
    # This is a placeholder - you need to implement proper tokenization
    # dataset = TextDataset(data_path, tokenizer, max_length=128)
    
    logger.info("Starting training...")
    # train(
    #     model=model,
    #     dataset=dataset,
    #     output_dir=full_output_dir,
    #     batch_size=batch_size,
    #     learning_rate=learning_rate,
    #     warmup_steps=warmup_steps,
    #     max_steps=max_steps,
    #     save_steps=save_steps
    # )
    
    logger.info("Training completed!")


def train_with_seeds(
    data_path: str,
    output_dir: str,
    seeds: List[int],
    **kwargs
):
    """Train LTG-BERT with multiple seeds"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path {data_path} does not exist!")
    
    if len(seeds) == 0:
        raise ValueError("At least one seed must be provided.")
    
    for seed in seeds:
        print("\n" + "=" * 80)
        print(f"Training with seed {seed}")
        print("=" * 80 + "\n")
        
        train_ltgbert(
            data_path=data_path,
            output_dir=output_dir,
            seed=seed,
            **kwargs
        )


# ============================================================================
# Command-Line Interface
# ============================================================================

def main():
    """Main function for command-line execution"""
    parser = argparse.ArgumentParser(
        description="Train LTG-BERT model for multilingual experiments"
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
        '--vocab_size',
        type=int,
        default=30000,
        help="Vocabulary size (default: 30000)"
    )
    parser.add_argument(
        '--hidden_size',
        type=int,
        default=768,
        help="Hidden dimension size (default: 768)"
    )
    parser.add_argument(
        '--num_layers',
        type=int,
        default=12,
        help="Number of transformer layers (default: 12)"
    )
    parser.add_argument(
        '--num_heads',
        type=int,
        default=12,
        help="Number of attention heads (default: 12)"
    )
    
    # Training configuration
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help="Training batch size (default: 8)"
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=100000,
        help="Maximum training steps (default: 100000)"
    )
    parser.add_argument(
        '--warmup_steps',
        type=int,
        default=10000,
        help="Warmup steps (default: 10000)"
    )
    parser.add_argument(
        '--save_steps',
        type=int,
        default=5000,
        help="Save checkpoint every N steps (default: 5000)"
    )
    
    args = parser.parse_args()
    
    # Train with all seeds
    train_with_seeds(
        data_path=args.data_path,
        output_dir=args.output_dir,
        seeds=args.seeds,
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps
    )


if __name__ == "__main__":
    main()