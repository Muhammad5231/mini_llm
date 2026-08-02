import math
import torch
from src.model import MiniLLM

def evaluate_model(model: MiniLLM, val_loader, device: str = "cpu") -> dict:
    """
    Computes Evaluation Metrics:
    1. Validation Loss: Average Cross-Entropy Loss over validation set.
    2. Perplexity (PPL): exp(Val Loss) - measure of distribution uncertainty.
    3. Token Prediction Accuracy: Percentage of correctly predicted next tokens.
    """
    model.eval()
    model.to(device)
    
    total_loss = 0.0
    correct_tokens = 0
    total_tokens = 0

    with torch.no_grad():
        for x_val, y_val, pad_mask in val_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            pad_mask = pad_mask.to(device) if pad_mask is not None else None

            logits, loss = model(x_val, y_val, pad_mask=pad_mask)
            total_loss += loss.item()

            # Calculate Accuracy
            predictions = torch.argmax(logits, dim=-1)
            mask = (y_val != -100) # Exclude padding indices
            correct_tokens += (predictions[mask] == y_val[mask]).sum().item()
            total_tokens += mask.sum().item()

    avg_loss = total_loss / len(val_loader)
    perplexity = math.exp(avg_loss)
    accuracy = (correct_tokens / max(1, total_tokens)) * 100.0

    return {
        "val_loss": round(avg_loss, 4),
        "perplexity": round(perplexity, 4),
        "accuracy_percent": round(accuracy, 2)
    }