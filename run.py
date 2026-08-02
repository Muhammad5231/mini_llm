import os
import argparse
import torch

from config import Config
from src.tokenizer import BPETokenizer
from src.dataset import prepare_data, load_all_txt_files
from src.model import MiniLLM
from src.train import train_model
from src.evaluate import evaluate_perplexity
from src.generate import interactive_chat

def main():
    parser = argparse.ArgumentParser(description="Mini LLM Version 2 Controller")
    parser.add_argument("--mode", type=str, default="train", choices=["train_tokenizer", "train", "eval", "chat"])
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    args = parser.parse_args()

    config = Config()
    
    # Set random seed for reproducibility
    torch.manual_seed(config.seed)

    if args.mode == "train_tokenizer":
        print("Training Byte-Pair Encoding (BPE) Tokenizer from scratch...")
        text_corpus = load_all_txt_files(config.data_dir)
        tokenizer = BPETokenizer(
            vocab_size=config.vocab_size,
            min_frequency=config.min_frequency,
            pad_token=config.pad_token,
            unk_token=config.unk_token,
            bos_token=config.bos_token,
            eos_token=config.eos_token
        )
        tokenizer.train(text_corpus)
        tokenizer.save(config.tokenizer_path)
        print(f"Tokenizer trained successfully! Saved to: {config.tokenizer_path}")

    elif args.mode == "train":
        if not os.path.exists(config.tokenizer_path):
            print("Tokenizer not found! Auto-training BPE tokenizer first...")
            os.system("python run.py --mode train_tokenizer")

        tokenizer = BPETokenizer.load(config.tokenizer_path)
        # Update config vocab_size to match actual tokenizer vocabulary
        config.vocab_size = len(tokenizer.encoder)

        train_loader, val_loader, total_tokens = prepare_data(config, tokenizer)
        model = MiniLLM(config)

        resume_path = config.latest_model_path if args.resume else None
        train_model(model, train_loader, val_loader, config, resume_path=resume_path)

    elif args.mode == "eval":
        if not os.path.exists(config.best_model_path):
            print("No trained checkpoint found. Please train the model first.")
            return

        tokenizer = BPETokenizer.load(config.tokenizer_path)
        config.vocab_size = len(tokenizer.encoder)

        _, val_loader, _ = prepare_data(config, tokenizer)
        
        # weights_only=False allows unpickling custom objects like Config safely from local files
        checkpoint = torch.load(config.best_model_path, map_location=config.device, weights_only=False)
        model = MiniLLM(config)
        model.load_state_dict(checkpoint["model_state_dict"])

        val_loss, perplexity = evaluate_perplexity(model, val_loader, config.device)
        print("\n" + "="*40)
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Perplexity (PPL): {perplexity:.4f}")
        print("="*40)

    elif args.mode == "chat":
        if not os.path.exists(config.best_model_path):
            print("No trained checkpoint found. Please train the model first.")
            return

        tokenizer = BPETokenizer.load(config.tokenizer_path)
        config.vocab_size = len(tokenizer.encoder)

        # weights_only=False allows unpickling custom objects like Config safely from local files
        checkpoint = torch.load(config.best_model_path, map_location=config.device, weights_only=False)
        model = MiniLLM(config)
        model.load_state_dict(checkpoint["model_state_dict"])

        interactive_chat(model, tokenizer, config)

if __name__ == "__main__":
    main()