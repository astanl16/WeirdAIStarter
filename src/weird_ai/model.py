"""GPT-style decoder-only language model for the Weird AI course.

This module assembles the transformer components implemented in earlier lessons.
Weights are randomly initialized; no pretrained checkpoint is loaded here.
"""

import torch
from torch import nn

from weird_ai.layer_norm import LayerNorm
from weird_ai.transformer import TransformerBlock


class WeirdAIModel(nn.Module):
    """Predict next-token vocabulary logits from batches of token IDs.

    Args:
        vocab_size: Number of tokens in the tokenizer vocabulary.
        emb_dim: Token and positional embedding dimension.
        context_length: Maximum number of tokens in a sequence.
        num_heads: Number of attention heads in each transformer block.
        num_layers: Number of transformer blocks.
        dropout: Dropout probability.
        qkv_bias: Whether attention projections use a bias.

    Input shape: (batch_size, sequence_length), containing integer token IDs.
    Output shape: (batch_size, sequence_length, vocab_size), containing logits.
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 128,
        context_length: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        qkv_bias: bool = False,
    ):
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if emb_dim <= 0 or context_length <= 0 or num_layers <= 0 or num_heads <= 0:
            raise ValueError("Embedding dimension, context length, heads, and layers must be positive")
        if emb_dim % num_heads != 0:
            raise ValueError("emb_dim must be divisible by num_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")

        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.context_length = context_length
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.position_embedding = nn.Embedding(context_length, emb_dim)
        self.embedding_dropout = nn.Dropout(dropout)

        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    emb_dim=emb_dim,
                    context_length=context_length,
                    num_heads=num_heads,
                    dropout=dropout,
                    qkv_bias=qkv_bias,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = LayerNorm(emb_dim=emb_dim)
        self.output_head = nn.Linear(emb_dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return unnormalized next-token scores for every input position."""
        if x.ndim != 2:
            raise ValueError("Expected token IDs with shape (batch_size, sequence_length)")

        _, sequence_length = x.shape
        if sequence_length > self.context_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds context length {self.context_length}"
            )

        positions = torch.arange(sequence_length, device=x.device)
        token_vectors = self.token_embedding(x)
        position_vectors = self.position_embedding(positions)
        hidden_states = self.embedding_dropout(token_vectors + position_vectors)

        hidden_states = self.transformer_blocks(hidden_states)
        hidden_states = self.final_norm(hidden_states)
        logits = self.output_head(hidden_states)
        return logits
