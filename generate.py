import torch
import torch.nn.functional as F
from tinyGPT import tinyGPT
from tokenization import encode, decode, vocab_size
from config import *
import os

def load_model(checkpoint_path):

    model =tinyGPT(
        vocab_size=vocab_size,
        d_model=EMBEDDING_DIM,
        max_len=MAX_SEQUENCE_LENGTH,
        num_heads=NUM_HEADS,
        num_transformer_blocks=NUM_TRANSFORMER_BLOCKS,
        d_ff=FF_DIM,
        dropout=0.0
    ).to(DEVICE)

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model from {checkpoint_path}")
    print(f"Validation loss: {checkpoint.get('val_loss', 'N/A')}")
    return model

@torch.no_grad()  
def generate(model, prompt, max_new_tokens=500, temperature=1.0, top_k=None, top_p=0.9):
    
    input_ids = encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)
    generated = []

    for _ in range(max_new_tokens):
        input_ids_cropped = input_ids if input_ids.size(1) <= MAX_SEQUENCE_LENGTH else input_ids[:, -MAX_SEQUENCE_LENGTH:]
        logits = model(input_ids_cropped)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        if top_p  < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('Inf')

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_id], dim=1)
        generated.append(next_id.item())
    
    full_text = decode(input_ids[0].tolist())
    return full_text