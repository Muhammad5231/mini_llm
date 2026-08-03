import os
import argparse
import json
import torch

from config import Config
from src.tokenizer import BPETokenizer
from src.dataset import prepare_data, load_all_txt_files, DatasetManager
from src.model import MiniLLM
from src.train import train_model
from src.evaluate import evaluate_model
from src.generate import interactive_chat_v4
from src.rag.rag import RAGPipeline
from src.api.api import start_api_server

def main():
    parser = argparse.ArgumentParser(description="Mini LLM Version 4 Master Controller")
    parser.add_argument("--mode", type=str, default="train", 
                        choices=["train_tokenizer", "train", "eval", "chat", "analyze", "rag", "api", "export", "import", "test"])
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    parser.add_argument("--query", type=str, default="", help="Query for RAG mode")
    parser.add_argument("--import_path", type=str, default="", help="Path for importing bundle")
    args = parser.parse_args()

    config = Config()
    torch.manual_seed(config.seed)

    if args.mode == "test":
        print("Running unit test suite for transformer components...")
        os.system("python -m unittest discover -s tests")

    elif args.mode == "train_tokenizer":
        print("Training Byte-Pair Encoding (BPE) Tokenizer recursively across data/...")
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

    elif args.mode == "analyze":
        if not os.path.exists(config.tokenizer_path):
            os.system("python run.py --mode train_tokenizer")
        tokenizer = BPETokenizer.load(config.tokenizer_path)
        manager = DatasetManager(config.data_dir, tokenizer)
        stats = manager.analyze()
        print("\n" + "="*50)
        print(" Dataset Analysis Summary (Version 4)")
        print("="*50)
        for k, v in stats.items():
            print(f"{k:<30}: {v}")
        print("="*50)

    elif args.mode == "train":
        if not os.path.exists(config.tokenizer_path):
            print("Tokenizer not found! Auto-training tokenizer first...")
            os.system("python run.py --mode train_tokenizer")

        tokenizer = BPETokenizer.load(config.tokenizer_path)
        config.vocab_size = len(tokenizer.encoder)

        train_loader, val_loader, total_tokens = prepare_data(config, tokenizer)
        model = MiniLLM(config)

        resume_path = config.latest_model_path if args.resume else None
        train_model(model, train_loader, val_loader, config, resume_path=resume_path)

    elif args.mode == "eval":
        if not os.path.exists(config.best_model_path):
            print("No checkpoint found. Please train model first.")
            return

        tokenizer = BPETokenizer.load(config.tokenizer_path)
        config.vocab_size = len(tokenizer.encoder)
        _, val_loader, _ = prepare_data(config, tokenizer)

        checkpoint = torch.load(config.best_model_path, map_location=config.device, weights_only=False)
        model = MiniLLM(config)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        metrics = evaluate_model(model, val_loader, config.device)
        print("\n" + "="*50)
        print(" Evaluation Metrics & Benchmarks")
        print("="*50)
        for k, v in metrics.items():
            print(f"{k:<25}: {v}")
        print("="*50)

    elif args.mode == "rag":
        rag = RAGPipeline(chunk_size=config.rag_chunk_size)
        rag.index_directory(config.data_dir)
        query = args.query if args.query else "What is machine learning?"
        print(f"\nQuerying RAG Vector DB for: '{query}'\n")
        context = rag.retrieve_context(query, top_k=2)
        print(context)

    elif args.mode == "api":
        if not os.path.exists(config.best_model_path):
            print("No trained checkpoint found. Please train the model first.")
            return

        tokenizer = BPETokenizer.load(config.tokenizer_path)
        config.vocab_size = len(tokenizer.encoder)

        checkpoint = torch.load(config.best_model_path, map_location=config.device, weights_only=False)
        model = MiniLLM(config)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        start_api_server(model, tokenizer, config)

    elif args.mode == "chat":
        if not os.path.exists(config.best_model_path):
            print("No trained checkpoint found. Please train the model first.")
            return

        tokenizer = BPETokenizer.load(config.tokenizer_path)
        config.vocab_size = len(tokenizer.encoder)

        checkpoint = torch.load(config.best_model_path, map_location=config.device, weights_only=False)
        model = MiniLLM(config)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        interactive_chat_v4(model, tokenizer, config)

    elif args.mode == "export":
        if not os.path.exists(config.best_model_path):
            print("No checkpoint found to export.")
            return
        
        export_bundle = {
            "model_checkpoint": config.best_model_path,
            "tokenizer": config.tokenizer_path,
            "config": config.__dict__
        }
        export_manifest = os.path.join(config.export_dir, "model_bundle.json")
        with open(export_manifest, "w", encoding="utf-8") as f:
            json.dump(export_bundle, f, indent=2)
        print(f"Model bundle exported successfully to: {config.export_dir}")

if __name__ == "__main__":
    main()