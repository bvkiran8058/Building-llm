import torch
from tokenization import vocab_size, vocab

BATCH_SIZE = 16
LEARNING_RATE = 3e-4
NUM_EPOCHS = 25
GRAD_CLIP = 1.0
EVAL_INTERVAL = 500
SAVE_INTERVAL = 1000
CHECKPOINT_DIR = './checkpoints'
EMBEDDING_DIM = 128
MAX_SEQUENCE_LENGTH = 512
VOCAB_SIZE = vocab_size
VOCAB = vocab
NUM_HEADS = 8
NUM_TRANSFORMER_BLOCKS = 6
FF_DIM = 512
DROPOUT_RATE = 0.1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'