import os
import time
import math
import torch
import matplotlib.pyplot as plt
from src.model import MiniLLM
from src.logger import TrainingLogger

def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, base_lr, min_lr):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        decayed_lr = min_lr + (base_lr - min_lr) * cosine_decay
        return decayed_lr / base_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_model(model: MiniLLM, train_loader, val_loader, config, resume_path: str = None):
    logger = TrainingLogger(config.log_dir, config.session_log_path)
    device = torch.device(config.device)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, config.warmup_epochs, config.epochs, config.learning_rate, config.min_lr
    )

    start_epoch = 0
    history = []
    best_val_loss = float("inf")
    patience_counter = 0

    # Auto Resume Checkpoint
    if resume_path and os.path.exists(resume_path):
        logger.log_info(f"Resuming checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        history = checkpoint.get("history", [])
        if history:
            best_val_loss = min(h["val_loss"] for h in history)

    summary = model.get_model_summary()
    logger.log_info(f"Summary: {summary['total_parameters']:,} Params | Est RAM: {summary['estimated_ram_mb']} MB")

    print("\n" + "="*80)
    print(f" Mini LLM Version 3 Training Dashboard | Device: {config.device.upper()}")
    print("="*80)
    print(f"{'Epoch':<8}{'Train Loss':<12}{'Val Loss':<12}{'LR':<12}{'Tok/sec':<12}{'ETA':<10}{'Elapsed':<10}")
    print("-" * 80)

    overall_start_time = time.time()

    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.time()
        model.train()
        total_train_loss = 0.0
        tokens_in_epoch = 0

        optimizer.zero_grad()

        for step, (x_b, y_b, pad_mask) in enumerate(train_loader):
            x_b, y_b = x_b.to(device), y_b.to(device)
            pad_mask = pad_mask.to(device) if pad_mask is not None else None

            # Scaled loss for Gradient Accumulation
            _, loss = model(x_b, y_b, pad_mask=pad_mask)
            loss = loss / config.grad_accum_steps
            loss.backward()

            total_train_loss += loss.item() * config.grad_accum_steps
            tokens_in_epoch += x_b.numel()

            # Step optimizer every grad_accum_steps batches
            if (step + 1) % config.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation Pass
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x_v, y_v, pad_mask_v in val_loader:
                x_v, y_v = x_v.to(device), y_v.to(device)
                pad_mask_v = pad_mask_v.to(device) if pad_mask_v is not None else None
                _, v_loss = model(x_v, y_v, pad_mask=pad_mask_v)
                total_val_loss += v_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # Timing metrics
        epoch_time = time.time() - epoch_start
        tokens_per_sec = int(tokens_in_epoch / max(0.001, epoch_time))
        elapsed_total = time.time() - overall_start_time
        remaining_epochs = config.epochs - (epoch + 1)
        eta_seconds = epoch_time * remaining_epochs

        # Log epoch record
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "learning_rate": round(current_lr, 6),
            "tokens_per_sec": tokens_per_sec,
            "elapsed_seconds": int(elapsed_total)
        }
        history.append(epoch_record)

        print(f"{epoch+1:02d}/{config.epochs:02d}   {avg_train_loss:<12.4f}{avg_val_loss:<12.4f}{current_lr:<12.6f}{tokens_per_sec:<12,}{int(eta_seconds)}s{'':<5}{int(elapsed_total)}s")

        # Checkpointing
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "config": config
        }
        torch.save(checkpoint_data, config.latest_model_path)

        # Early Stopping & Best Checkpoint Tracking
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(checkpoint_data, config.best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print(f"\n[Early Stopping Triggered] Validation loss did not improve for {config.early_stopping_patience} consecutive epochs.")
                break

    # Save Logs & History
    logger.save_history_json(history, config.history_json_path)
    logger.save_history_csv(history, config.history_csv_path)

    # Plot Loss Curve
    plt.figure(figsize=(8, 5))
    epochs_range = [h["epoch"] for h in history]
    plt.plot(epochs_range, [h["train_loss"] for h in history], label="Train Loss", color="blue")
    plt.plot(epochs_range, [h["val_loss"] for h in history], label="Val Loss", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Mini LLM Version 3 Loss Dashboard")
    plt.legend()
    plt.grid(True)
    plt.savefig(config.loss_plot_path)
    plt.close()

    print("="*80)
    print(f"Training session complete! History saved to {config.log_dir}")