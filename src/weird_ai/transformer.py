from torch import nn as nn
# found out this better way for import statements 
from .attention import CausalAttention
from .layer_norm import LayerNorm
from .feed_forward import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, context_length, num_heads=12, dropout=0.1, qkv_bias=False):
        super().__init__()

        self.norm1 = LayerNorm(emb_dim)
        self.attn = CausalAttention(
            embedding_dim=emb_dim,
            output_dim=emb_dim,
            context_length=context_length,
            dropout=dropout,
            qkv_bias=qkv_bias,
        )
        self.norm2 = LayerNorm(emb_dim)
        self.ff = FeedForward(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x