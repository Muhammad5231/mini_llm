import time
import torch
import matplotlib.pyplot as plt
from src.model import MiniLLM

def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, base_lr, min_lr):
    """
    Cosine Annealing Learning Rate Scheduler with Linear Warmup.
    
    Equations:
    Warmup (epoch <= warmup_epochs):
        lr = base_lr * (epoch / warmup_epochs)
    Cosine Annealing (epoch > warmup_epochs):
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * progress))
    """
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        decayed_lr = min_lr + (base_lr - min_lr) * cosine_decay
        return decayed_lr / base_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

import math

def train_model(model: MiniLLM, train_loader, val_loader, config, resume_path: str = None):
    """Complete Training Loop with metrics logging, checkpointing, and graph plotting."""
    device = torch.device(config.device)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, config.warmup_epochs, config.epochs, config.learning_rate, config.min_lr
    )

    start_epoch = 0
    train_losses = []
    val_losses = []

    # Automatic Checkpoint Resuming
    if resume_path and torch.os.path.exists(resume_path):
        print(f"Resuming training from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        train_losses = checkpoint.get("train_losses", [])
        val_losses = checkpoint.get("val_losses", [])

    print("\n" + "="*70)
    summary = model.get_model_summary()
    print(f"Model Summary: {summary['total_parameters']:,} Parameters | Size: {summary['model_size_mb']} MB")
    print(f"Device: {config.device.upper()} | Training Epochs: {config.epochs}")
    print("="*70)
    print(f"{'Epoch':<8}{'Train Loss':<14}{'Val Loss':<14}{'LR':<12}{'ETA':<10}{'Tokens':<12}")
    print("-" * 70)

    start_time = time.time()
    total_tokens_processed = 0

    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.time()
        model.train()
        total_train_loss = 0.0

        for x_b, y_b, pad_mask in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            pad_mask = pad_mask.to(device) if pad_mask is not None else None

            optimizer.zero_grad()
            _, loss = model(x_b, y_b, pad_mask)
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            optimizer.step()
            total_train_loss += loss.item()
            total_tokens_processed += x_b.numel()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation Loop
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x_v, y_v, pad_mask_v in val_loader:
                x_v, y_v = x_v.to(device), y_v.to(device)
                pad_mask_v = pad_mask_v.to(device) if pad_mask_v is not None else None
                _, v_loss = model(x_v, y_v, pad_mask_v)
                total_val_loss += v_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Update Scheduler
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # ETA Calculation
        elapsed = time.time() - epoch_start
        remaining_epochs = config.epochs - (epoch + 1)
        eta_seconds = elapsed * remaining_epochs
        eta_str = f"{int(eta_seconds)}s"

        print(f"{epoch+1:02d}/{config.epochs:02d}   {avg_train_loss:<14.4f}{avg_val_loss:<14.4f}{current_lr:<12.6f}{eta_str:<10}{total_tokens_processed:<12,}")

        # Save Checkpoints
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "config": config
        }
        torch.save(checkpoint_data, config.latest_model_path)

        if avg_val_loss == min(val_losses):
            torch.save(checkpoint_data, config.best_model_path)

    # Plot Loss Graph
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", color="blue")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Val Loss", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Mini LLM Version 2 Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(config.loss_plot_path)
    plt.close()

    print("="*70)
    print(f"Training Complete! Saved loss graph to: {config.loss_plot_path}")