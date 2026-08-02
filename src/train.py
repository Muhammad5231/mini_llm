import os
import torch
from src.model import MiniLLM

def train_model(
    model: MiniLLM,
    train_loader,
    val_loader,
    epochs: int = 15,
    lr: float = 1e-3,
    save_path: str = "mini_llm.pt"
):
    """Training loop optimized for CPU performance."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    print(f"Starting training on CPU... Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits, loss = model(x_batch, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation Phase
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                _, val_loss = model(x_val, y_val)
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch {epoch + 1:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Save model weights and metadata
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            "vocab_size": model.tok_emb.num_embeddings,
            "n_embd": model.tok_emb.embedding_dim,
            "n_layer": len(model.blocks),
            "n_head": model.blocks[0].attn.n_head,
            "block_size": model.block_size
        }
    }
    torch.save(checkpoint, save_path)
    print(f"\nModel successfully saved to {save_path}")