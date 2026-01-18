"""
embeddings and positional encodings for the transformer model

embeddings_dim = 32
vocab_size = 81

embedding_layer = [
    [
    random 32 weights
    ],
    [
    random 32 weights
    ],
    [
    random 32 weights
    ],          ........................ with 81 rows
]

shape = (81, 32)

positional encodings
max_len = 5000
positions = [
    [0],
    [0],
    [0],
    [0],
    ................... with 5000 rows
]

div_term = 1 / (10000 ^ (2i / d_model)) as per attention is all you need paper
with 32 embedding dimensions we will get 16 div terms

[1.0000e+00, 5.6234e-01, 3.1623e-01, 1.7783e-01, 1.0000e-01, 5.6234e-02,
3.1623e-02, 1.7783e-02, 1.0000e-02, 5.6234e-03, 3.1623e-03, 1.7783e-03,
1.0000e-03, 5.6234e-04, 3.1623e-04, 1.7783e-04]

now encoding the positions with sin for for even places and cos for odd places

positional encoding matrix will look like 

[[ 0.0000e+00,  1.0000e+00,  0.0000e+00,  ...,  1.0000e+00,
0.0000e+00,  1.0000e+00],
[ 8.4147e-01,  5.4030e-01,  5.3317e-01,  ...,  1.0000e+00,
1.7783e-04,  1.0000e+00],
[ 9.0930e-01, -4.1615e-01,  9.0213e-01,  ...,  1.0000e+00,
3.5566e-04,  1.0000e+00],
...,
[ 3.7961e-01, -9.2515e-01, -9.0865e-01,  ...,  9.9953e-01,
1.7248e-02,  9.9985e-01],
[-5.7338e-01, -8.1929e-01, -9.9136e-01,  ...,  9.9952e-01,
1.7426e-02,  9.9985e-01],
[-9.9921e-01,  3.9821e-02, -7.6875e-01,  ...,  9.9951e-01,
1.7604e-02,  9.9985e-01]])............... with size (5000, 32)

example:
let's say we have a sentence of length 20
embeddings shape = (20, 32)
we will add batch dimension to make it (1, 20, 32)
we will slice positional encoding matrix to (1, 20, 32)
and add it to the embeddings
final shape will be (1, 20, 32)
"""

import math
from tokenization import encode, decode
import torch
import torch.nn as nn
from config import *

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        Args:
            d_model: The dimension of the embeddings.
            max_len: The maximum sequence length your model will handle.
            dropout: Dropout probability.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        """
        
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    
embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=EMBEDDING_DIM)
pos_encoder = PositionalEncoding(d_model=EMBEDDING_DIM, max_len=MAX_SEQUENCE_LENGTH)

text = "Bhishma is very wise"
tokens = torch.tensor(encode(text), dtype=torch.long)
print(f"tokens: {tokens}")

embeddings = embedding_layer(tokens)
print(f"embeddings.shape: {embeddings.shape}")

embeddings_with_batch = embeddings.unsqueeze(0)
print(f"embeddings_with_batch.shape: {embeddings_with_batch.shape}")

x = embeddings_with_batch * math.sqrt(EMBEDDING_DIM)
print(f"x.shape: {x.shape}")

x = pos_encoder(x)
print(f"x.shape after positional encoding: {x.shape}")