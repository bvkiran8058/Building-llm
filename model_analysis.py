from tinyGPT import tinyGPT
from config import *

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def analyze_model():
    model = tinyGPT(
        vocab_size=VOCAB_SIZE,
        d_model=EMBEDDING_DIM,
        max_len=MAX_SEQUENCE_LENGTH,
        num_heads=NUM_HEADS,
        num_transformer_blocks=NUM_TRANSFORMER_BLOCKS,
        d_ff=FF_DIM,
        dropout=0.1
    )

    total_params, trainable_params = count_parameters(model)
    print("=" * 50)
    print("TinyGPT architecture Analysis")
    print("=" * 50)

    print(f"\nConfiguration:")
    print(f"Vocab Size: {VOCAB_SIZE}")
    print(f"Embedding Dim: {EMBEDDING_DIM}")
    print(f"Max Sequence Length: {MAX_SEQUENCE_LENGTH}")
    print(f"Number of Heads: {NUM_HEADS}")
    print(f"Number of Transformer Blocks: {NUM_TRANSFORMER_BLOCKS}")
    print(f"Feedforward Dim: {FF_DIM}")
    print(f"Dropout Rate: {DROPOUT_RATE}")

    print(f"\nModel Parameters:")
    print(f"Total Parameters: {total_params}, ({total_params/1e6:.2f} Million)")
    print(f"Trainable Parameters: {trainable_params}, ({trainable_params/1e6:.2f} Million)")

analyze_model()