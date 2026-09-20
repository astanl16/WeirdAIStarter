import torch
import torch.nn as nn

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gelu = nn.GELU(approximate='tanh')

    def forward(self, x):
        return self.gelu(x)
    
    
class FeedForward(nn.Module):

    def __init__(self, emb_dim):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
        )

    def forward(self, x):
        return self.layers(x)