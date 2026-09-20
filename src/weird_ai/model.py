import torch
import torch.nn as nn


class WeirdAIModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=4,
                 num_layers=2, context_size=128, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_size = context_size

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(context_size, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        B, T = x.shape
        if T > self.context_size:
            x = x[:, -self.context_size:]
            T = self.context_size

        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.token_embedding(x) + self.position_embedding(pos)

        causal = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        h = self.transformer(h, mask=causal)

        return self.lm_head(h)