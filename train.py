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

    if not os.path.exists(CHECKPOINT_DIR):
        return None
    
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('checkpoint_step_')]
    if not checkpoints:
        return None
    
    steps = [int(f.split('_')[-1].replace('.pt', '')) for f in checkpoints]
    latest_step = max(steps)
    latest_checkpoint = os.path.join(CHECKPOINT_DIR, f'checkpoint_step_{latest_step}.pt')
    return latest_checkpoint

class CharDataset(Dataset):
    def __init__(self, text, block_size):
        self.data = torch.tensor(encode(text), dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size+1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

def get_batch(loader):

    try:
        return next(loader)
    except StopIteration:
        return None
    
@torch.no_grad()
def estimate_loss(model, loader, eval_iters=100):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        batch = get_batch(loader)
        if batch is None:
            break
        x, y = batch
        x, y = x.to(DEVICE), y.to(DEVICE)

        logits = model(x)
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses) if losses else float('inf')

def train():

    n = len(corpus)
    train_data = corpus[:int(0.9*n)]
    val_data = corpus[int(0.9*n):]

    train_dataset = CharDataset(train_data, MAX_SEQUENCE_LENGTH)
    val_dataset = CharDataset(val_data, MAX_SEQUENCE_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True if DEVICE=='cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True if DEVICE=='cuda' else False)

    model = tinyGPT(
        vocab_size=VOCAB_SIZE,
        d_model=EMBEDDING_DIM,
        max_len=MAX_SEQUENCE_LENGTH,
        num_heads=NUM_HEADS,
        num_transformer_blocks=NUM_TRANSFORMER_BLOCKS,
        d_ff=FF_DIM,
        dropout=DROPOUT_RATE
    ).to(DEVICE)

    print(f"Model has {sum(p.numel() for p in model.parameters())/1e6:.2f} million parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1, betas=(0.9, 0.95))

    warmup_steps = 500
    max_steps = NUM_EPOCHS * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step/warmup_steps
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    batches_per_epoch = len(train_loader)
    start_batch = 0

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

        print(f"Resumed from checkpoint: {latest_checkpoint} at epoch {start_epoch}, step {global_step}")
    
    else:
        print("No checkpoint found, starting training from scratch.")
    
    model.train()

    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

        for batch_idx, (x, y) in enumerate(progress_bar):

            if epoch == start_epoch and batch_idx < start_batch:
                continue

            x, y = x.to(DEVICE), y.to(DEVICE)

            logits = model(x)
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            progress_bar.set_postfix({'Loss': loss.item(), 'LR': scheduler.get_last_lr()[0]})

            if global_step % EVAL_INTERVAL == 0:
                val_loss = estimate_loss(model, val_loader)
                print(f"\nStep {global_step}: Validation Loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_path = os.path.join(CHECKPOINT_DIR, f'best_model.pt')
                    torch.save({
                        'epoch': epoch,
                        'batch_idx': batch_idx,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_loss': best_val_loss,
                    }, checkpoint_path)
                    print(f"Best model saved with validation loss: {best_val_loss:.4f}")

            if global_step % SAVE_INTERVAL == 0:
                checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_step_{global_step}.pt')
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': best_val_loss,
                    'batch_idx': batch_idx
                }, checkpoint_path)
                print(f"Checkpoint saved at step {global_step}")

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f}")

        start_batch = 0
    print("Training completed.")

if __name__ == "__main__":
    train()