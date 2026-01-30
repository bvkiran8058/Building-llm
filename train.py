"""
Corpus (raw text)
↓
Character-level tokenization
↓
Encoded as integers (vocab_size = 81)
↓
Sliding window of length = MAX_SEQUENCE_LENGTH + 1
↓
Input  x : (batch_size, 20)
Target y : (batch_size, 20)

x → tinyGPT
------------------------------------
Embedding:
(batch, 20) → (batch, 20, 32)
× sqrt(d_model)

+ Positional Encoding
(batch, 20, 32)

Transformer Blocks (causal masked attention)
(batch, 20, 32) → (batch, 20, 32)

LayerNorm
(batch, 20, 32)

Linear Head (weight-tied with embedding)
(batch, 20, 32) @ (32, vocab_size)
→ logits: (batch, 20, 81)
------------------------------------

Loss:
CrossEntropy over all tokens
((batch × 20), vocab_size) vs ((batch × 20))

Backpropagation
↓
Gradient clipping
↓
AdamW optimizer step
↓
Cosine LR schedule with warmup
↓
Checkpoint save / resume

"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tinyGPT import tinyGPT
from tokenization import encode, decode, vocab_size, corpus
from config import *
import os, math
from tqdm import tqdm

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def get_latest_checkpoint():
    """
    Finds the checkpoint with the largest training step.
    Used to resume training if interrupted.
    """
    if not os.path.exists(CHECKPOINT_DIR):
        return None
    
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith('checkpoint_step_')]
    if not checkpoints:
        return None
    
    steps = [int(f.split('_')[-1].replace('.pt', '')) for f in checkpoints]
    latest_step = max(steps)
    
    return os.path.join(CHECKPOINT_DIR, f'checkpoint_step_{latest_step}.pt')


class CharDataset(Dataset):
    """
    Creates sliding windows over character-tokenized text.

    For block_size = 20:
    input  x = [c1, c2, ..., c20]
    target y = [c2, c3, ..., c21]
    """
    def __init__(self, text, block_size):
        self.data = torch.tensor(encode(text), dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        # One-step-ahead prediction → need block_size + 1
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1]   # input sequence
        y = chunk[1:]    # next-token targets
        return x, y


def get_batch(loader):
    """
    Safely fetches the next batch from a DataLoader iterator.
    """
    try:
        return next(loader)
    except StopIteration:
        return None

    
@torch.no_grad()
def estimate_loss(model, loader, eval_iters=100):
    """
    Computes average validation loss over eval_iters batches.
    """
    model.eval()
    losses = []

    for _ in range(eval_iters):
        batch = get_batch(loader)
        if batch is None:
            break

        x, y = batch
        x, y = x.to(DEVICE), y.to(DEVICE)

        logits = model(x)  # (batch, seq_len, vocab_size)

        # Flatten for cross-entropy
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses) if losses else float('inf')

def train():
    """
    Main training loop for character-level GPT.
    Handles:
    - dataset creation
    - model initialization
    - optimizer + scheduler
    - checkpoint resume
    - training, validation, saving
    """

    # ---------------------------
    # 1. Train / Validation Split
    # ---------------------------
    n = len(corpus)
    train_data = corpus[:int(0.9 * n)]
    val_data   = corpus[int(0.9 * n):]

    print(f"[DEBUG] Corpus length: {n}")
    print(f"[DEBUG] Train chars: {len(train_data)}, Val chars: {len(val_data)}")

    # ---------------------------
    # 2. Dataset & DataLoader
    # ---------------------------
    train_dataset = CharDataset(train_data, MAX_SEQUENCE_LENGTH)
    val_dataset   = CharDataset(val_data, MAX_SEQUENCE_LENGTH)

    print(f"[DEBUG] Train samples: {len(train_dataset)}")
    print(f"[DEBUG] Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True if DEVICE == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if DEVICE == 'cuda' else False
    )

    # ---------------------------
    # 3. Model Initialization
    # ---------------------------
    model = tinyGPT(
        vocab_size=VOCAB_SIZE,
        d_model=EMBEDDING_DIM,
        max_len=MAX_SEQUENCE_LENGTH,
        num_heads=NUM_HEADS,
        num_transformer_blocks=NUM_TRANSFORMER_BLOCKS,
        d_ff=FF_DIM,
        dropout=DROPOUT_RATE
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model parameters: {total_params/1e6:.2f}M")
    print(f"[INFO] Training on device: {DEVICE}")

    # ---------------------------
    # 4. Optimizer & Scheduler
    # ---------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.1,
        betas=(0.9, 0.95)
    )

    warmup_steps = 500
    max_steps = NUM_EPOCHS * len(train_loader)

    print(f"[DEBUG] Max training steps: {max_steps}")
    print(f"[DEBUG] Warmup steps: {warmup_steps}")

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---------------------------
    # 5. Training State
    # ---------------------------
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    batches_per_epoch = len(train_loader)
    start_batch = 0

    # ---------------------------
    # 6. Resume from Checkpoint
    # ---------------------------
    latest_checkpoint = get_latest_checkpoint()

    if latest_checkpoint:
        checkpoint = torch.load(latest_checkpoint, map_location=DEVICE)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        start_batch = checkpoint.get('batch_idx', 0) + 1

        if start_batch >= batches_per_epoch:
            start_epoch += 1
            start_batch = 0

        print(f"[INFO] Resumed from checkpoint: {latest_checkpoint}")
        print(f"[INFO] Start epoch: {start_epoch}, Global step: {global_step}")
        print(f"[INFO] Best val loss so far: {best_val_loss:.4f}")

    else:
        print("[INFO] No checkpoint found. Starting from scratch.")

    model.train()

    # ---------------------------
    # 7. Training Loop
    # ---------------------------
    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_loss = 0.0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"
        )

        for batch_idx, (x, y) in enumerate(progress_bar):

            # Skip already-trained batches when resuming
            if epoch == start_epoch and batch_idx < start_batch:
                continue

            x, y = x.to(DEVICE), y.to(DEVICE)

            # DEBUG: shape sanity check (first batch only)
            if global_step == 0:
                print(f"[DEBUG] Input shape: {x.shape}")
                print(f"[DEBUG] Target shape: {y.shape}")

            # ---------------------------
            # Forward pass
            # ---------------------------
            logits = model(x)

            # DEBUG: logits shape
            if global_step == 0:
                print(f"[DEBUG] Logits shape: {logits.shape}")

            # ---------------------------
            # Loss computation
            # ---------------------------
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )

            # ---------------------------
            # Backpropagation
            # ---------------------------
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (important for transformers)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), GRAD_CLIP
            )

            # DEBUG: gradient norm
            if global_step % 200 == 0:
                print(f"[DEBUG] Grad norm: {grad_norm:.4f}")

            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'LR': f"{scheduler.get_last_lr()[0]:.6f}"
            })

            # ---------------------------
            # Validation
            # ---------------------------
            if global_step % EVAL_INTERVAL == 0:
                val_loss = estimate_loss(model, val_loader)
                print(f"\n[VAL] Step {global_step} | Val Loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        'epoch': epoch,
                        'batch_idx': batch_idx,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_loss': best_val_loss
                    }, os.path.join(CHECKPOINT_DIR, 'best_model.pt'))

                    print(f"[INFO] New best model saved (val_loss={best_val_loss:.4f})")

            # ---------------------------
            # Periodic Checkpoint
            # ---------------------------
            if global_step % SAVE_INTERVAL == 0:
                torch.save({
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': best_val_loss
                }, os.path.join(
                    CHECKPOINT_DIR,
                    f'checkpoint_step_{global_step}.pt'
                ))

                print(f"[INFO] Checkpoint saved at step {global_step}")

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"[INFO] Epoch {epoch+1} completed | Avg Loss: {avg_epoch_loss:.4f}")

        start_batch = 0  # reset after resume epoch

    print("[INFO] Training completed successfully.")


if __name__ == "__main__":
    train()