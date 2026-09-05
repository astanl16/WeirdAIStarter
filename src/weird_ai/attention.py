import torch
import torch.nn as nn


class SimpleSelfAttention(nn.Module):
    """
    A simple self-attention module without trainable query, key, and value projections.

    This version is primarily for learning.
    """

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (num_tokens, embedding_dim)

        Returns:
            context_vectors: Tensor of shape (num_tokens, embedding_dim)
            attention_weights: Tensor of shape (num_tokens, num_tokens)
        """

        # TODO:
        # 1. Compute attention scores using matrix multiplication.
        attention_scores = x @ x.T
        # 2. Normalize scores with softmax.
        attention_weights = torch.softmax(attention_scores, dim=-1)
        # 3. Compute context vectors as weighted sums of input vectors.
        context_vectors = attention_weights @ x

        return context_vectors, attention_weights

class SelfAttention(nn.Module):
    """
    Trainable self-attention using query, key, and value projections.
    """

    def __init__(self, embedding_dim, output_dim, qkv_bias=False):
        super().__init__()

        self.query = nn.Linear(embedding_dim, output_dim, bias=qkv_bias)
        self.key = nn.Linear(embedding_dim, output_dim, bias=qkv_bias)
        self.value = nn.Linear(embedding_dim, output_dim, bias=qkv_bias)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (num_tokens, embedding_dim)

        Returns:
            context_vectors: Tensor of shape (num_tokens, output_dim)
            attention_weights: Tensor of shape (num_tokens, num_tokens)
        """

        # TODO:
        # 1. Compute queries, keys, and values.
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        # 2. Compute scaled attention scores.
        attention_scores = queries @ keys.T / (keys.shape[-1] ** 0.5)

        # 3. Apply softmax.
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # 4. Compute context vectors.
        context_vectors = attention_weights @ values

        return context_vectors, attention_weights

class CausalAttention(nn.Module):
    """
    Self-attention with a causal mask so tokens cannot attend to future tokens.
    """

    def __init__(self, embedding_dim, output_dim, context_length, dropout=0.0, qkv_bias=False):
        super().__init__()

        self.query = nn.Linear(embedding_dim, output_dim, bias=qkv_bias)
        self.key = nn.Linear(embedding_dim, output_dim, bias=qkv_bias)
        self.value = nn.Linear(embedding_dim, output_dim, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, num_tokens, embedding_dim)

        Returns:
            context_vectors: Tensor of shape (batch_size, num_tokens, output_dim)
        """

        batch_size, num_tokens, embedding_dim = x.shape

        # TODO:
        # 1. Compute keys, queries, and values.
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        # 2. Compute scaled attention scores.
        attention_scores = queries @ keys.transpose(-2, -1) / (keys.shape[-1] ** 0.5)

        # 3. Mask future tokens.
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attention_scores = attention_scores.masked_fill(mask_bool, -torch.inf)

        # 4. Apply softmax.
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # 5. Apply dropout.
        attention_weights = self.dropout(attention_weights)

        # 6. Compute context vectors.
        context_vectors = attention_weights @ values

        return context_vectors