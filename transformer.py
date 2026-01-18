"""
This comes Just after Sinusoidal positional encoding (Attention is all you need)

This transformer block contains multi attention block, normalization and feed forward blocks

text: "Bhishma is very wise" (20 chars)
input (x): (batch_size, seq_length, embed dimensions) (1, 20, 32)
Block Config: d_model=32, num_heads=4, d_ff=128

residual = x shape (1, 20, 32)

x given to self attention to get the context aware vector of size (1, 20, 32)

now we will do residual + x  and norm(residual + x)

again we will preserve residual = x

x given to feed forward block will outputs (1, 20, 32)

now we will again do residual + x  and norm(residual + x)
"""


import torch
import torch.nn as nn
from attention import MultiHeadAttention

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=512, dropout=0.1):
        
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) 
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model) 

    def forward(self, x):
        
        # x = torch.relu(self.linear1(x))
        x = nn.functional.gelu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm1 = nn.LayerNorm(d_model)

        self.feed_forward = FeedForward(d_model, d_ff)
        self.dropout2 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        
        residual = x 
        
        x, _ = self.self_attn(q=x, k=x, v=x, mask=mask)
        x = self.dropout1(x)

        x = self.norm1(x + residual)

        residual = x
        
        x = self.feed_forward(x)
        x = self.dropout2(x)
        
        x = self.norm2(x + residual)
        
        return x