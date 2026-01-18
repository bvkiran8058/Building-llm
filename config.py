import torch
from tokenization import vocab_size, vocab

EMBEDDING_DIM = 128
MAX_SEQUENCE_LENGTH = 5000
VOCAB_SIZE = vocab_size
VOCAB = vocab
NUM_HEADS = 8
NUM_TRANSFORMER_BLOCKS = 6
FF_DIM = 512
DROPOUT_RATE = 0.1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'