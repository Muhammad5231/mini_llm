import math
import torch
from src.model import MiniLLM

def evaluate_perplexity(model: MiniLLM, val_loader, device: str = "cpu") -> tuple[float, float]:
    """
    Evaluates Validation Loss and Perplexity (PPL).
    
    Mathematical Formula:
    Perplexity = exp( CrossEntropyLoss )
    PPL measures how well the probability distribution predicts the sample.
    Lower perplexity indicates better performance.
    """
    model.eval()
    model.to(device)
    total_loss = 0.0

    with torch.no_grad():
        for x_val, y_val, pad_mask in val_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            pad_mask = pad_mask.to(device) if pad_mask is not None else None
            
            _, loss = model(x_val, y_val, pad_mask)
            total_loss += loss.item()

    avg_val_loss = total_loss / len(val_loader)
    perplexity = math.exp(avg_val_loss)

    return avg_val_loss, perplexity