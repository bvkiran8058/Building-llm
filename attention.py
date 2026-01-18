"""
Multi head attention block with let's with 4 heads

first we will create matrix for wq, wk, wv with size (embed_dim,embed_dim)
and also combined weights

text: "Bhishma is very wise" (20 chars)
x = (1, 20,32)

we will give these x to wq, wk, wv parallely to generate Q, K, V with each shape= (1, 20, 32)

now we will divide embedding dimensions with heads to split for each head
gets shape (1, 20, 4, 8) now transpose to (1, 4, 20, 8)

Now we will use attention dot product formula to get attention weights
Attention(Q, K, V) = softmax((Q.K^T)/sqrt(head_dim)) * V

    Q: (1, 4, 20, 8)
    K Transposed: (1, 4, 8, 20)
    Operation: Q @ K.T
    Output attn_scores: (1, 4, 20, 20)

    and divide by sqrt(8) to stabilize gradients

masking : we set the upper triangle of the 20x20 grid to -infinity.

Softmax:
    Operation: Convert scores to probabilities (0.0 to 1.0).
    Output attn_weights: (1, 4, 20, 20)

Concatenate Heads:
    We put the pieces back together
    Transpose Back: (1, 4, 20, 8) to (1, 20, 4, 8)
    Flatten (.view): (1, 20, 4, 8) to (1, 20, 32)

Again we will send this to another layer w_o which will mix all heads
"""

import torch, math
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model 
        self.num_heads = num_heads 
        self.head_dim = d_model // num_heads 

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.w_q = nn.Linear(d_model, d_model) 
        self.w_k = nn.Linear(d_model, d_model) 
        self.w_v = nn.Linear(d_model, d_model) 

        self.w_o = nn.Linear(d_model, d_model) 

    def scaled_dot_product_attention(self, Q, K, V, mask=None):

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) 

        attn_scores = attn_scores / math.sqrt(self.head_dim)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)
        
        return output, attn_weights

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        Q = self.w_q(q)
        K = self.w_k(k)
        V = self.w_v(v)

        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, -1, self.d_model)

        return self.w_o(output)