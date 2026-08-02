import os
import argparse
from src.dataset import prepare_dataloaders, CharTokenizer
from src.model import MiniLLM
from src.train import train_model
from src.generate import interactive_cli

def main():
    parser = argparse.ArgumentParser(description="Train or Run Educational Mini Transformer")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "chat"], help="Mode: train or chat")
    args = parser.parse_args()

    data_path = os.path.join("data", "input.txt")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing text dataset file at {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Hyperparameters as requested for Version 1
    BLOCK_SIZE = 64
    BATCH_SIZE = 16
    N_EMBD = 64
    N_LAYER = 2
    N_HEAD = 2
    EPOCHS = 20

    train_loader, val_loader, tokenizer = prepare_dataloaders(
        text, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE
    )

    if args.mode == "train":
        model = MiniLLM(
            vocab_size=tokenizer.vocab_size,
            n_embd=N_EMBD,
            n_layer=N_LAYER,
            n_head=N_HEAD,
            block_size=BLOCK_SIZE
        )
        train_model(model, train_loader, val_loader, epochs=EPOCHS, save_path="mini_llm.pt")

    elif args.mode == "chat":
        interactive_cli("mini_llm.pt", tokenizer)

if __name__ == "__main__":
    main()