import math
import time
import torch
import psutil
from src.model import MiniLLM

def evaluate_model(model: MiniLLM, val_loader, device: str = "cpu") -> dict:
    """
    Evaluates Validation Loss, Perplexity, Token Prediction Accuracy, CPU & RAM footprint.
    """
    model.eval()
    model.to(device)
    
    total_loss = 0.0
    correct_tokens = 0
    total_tokens = 0

    start_time = time.time()
    process = psutil.Process()

    with torch.no_grad():
        for x_val, y_val, pad_mask in val_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            pad_mask = pad_mask.to(device) if pad_mask is not None else None

            logits, loss = model(x_val, y_val, pad_mask=pad_mask)
            total_loss += loss.item()

            predictions = torch.argmax(logits, dim=-1)
            mask = (y_val != -100)
            correct_tokens += (predictions[mask] == y_val[mask]).sum().item()
            total_tokens += mask.sum().item()

    elapsed = time.time() - start_time
    avg_loss = total_loss / len(val_loader)
    perplexity = math.exp(avg_loss)
    accuracy = (correct_tokens / max(1, total_tokens)) * 100.0

    # System resource benchmarking
    ram_usage_mb = process.memory_info().rss / (1024 * 1024)
    cpu_percent = psutil.cpu_percent(interval=0.1)

    return {
        "val_loss": round(avg_loss, 4),
        "perplexity": round(perplexity, 4),
        "accuracy_percent": round(accuracy, 2),
        "benchmark_eval_time_sec": round(elapsed, 3),
        "ram_usage_mb": round(ram_usage_mb, 2),
        "cpu_percent": cpu_percent
    }