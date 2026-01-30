"""
text: "Bhishma is very wise" (20 chars)

first we will generate for seq_lenght shape (20, 20)

now we will generate embeddings for characters 20 output shape: (1, 20, 32) and multiply each value with sqrt(dim)

we will add those embeddings with positional encodings to store positions (1, 20, 32)

attention blocks (1, 20, 32) --> (1, 20, 32)

again we will normalize them 

last exit : (1, 20, 32) @ (32, 81) to return output shape with vocab size
(1, 20, 81)

Now for every single 20 positions have 81 scores

"""


import torch
import torch.nn as nn
from transformer import TransformerBlock
from embeddings import PositionalEncoding
import math

def generate_square_subsequent_mask(sz):
    """
    Returns mask of shape (1, 1, sz, sz)
    1 = allowed
    0 = blocked
    """
    mask = torch.tril(torch.ones(sz, sz))
    return mask.unsqueeze(0).unsqueeze(0)


class tinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_transformer_blocks, d_ff, max_len, dropout=0.1):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_transformer_blocks)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        batch_size, seq_len = idx.shape
        
        mask = generate_square_subsequent_mask(seq_len).to(idx.device)
        
        x = self.token_embedding(idx) * math.sqrt(self.token_embedding.embedding_dim)
        x = self.positional_encoding(x)
        
        for block in self.blocks:
            x = block(x, mask=mask)
            
        x = self.ln_f(x)
        
        logits = self.lm_head(x)
        
        return logits