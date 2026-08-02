import os
import re
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Any
from src.tokenizer import BPETokenizer

def clean_text(text: str) -> str:
    """Removes control characters while preserving standard printable text, newlines, and punctuation."""
    text = re.sub(r'[^\x20-\x7E\n\t]', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def load_all_txt_files(data_dir: str) -> str:
    """Scans data directory, cleans and aggregates text from all .txt files."""
    files = glob.glob(os.path.join(data_dir, "*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in directory: {data_dir}")
    
    combined = []
    for fpath in sorted(files):
        with open(fpath, "r", encoding="utf-8") as f:
            raw = f.read()
            cleaned = clean_text(raw)
            if cleaned:
                combined.append(cleaned)
            
    return "\n\n".join(combined)


class DynamicTextDataset(Dataset):
    """PyTorch Dataset extracting sequence chunks for next-token prediction."""
    def __init__(self, token_ids: List[int], block_size: int):
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


def pad_collate_fn(batch, pad_idx: int):
    """Pads short sequences dynamically and generates pad mask."""
    x_list, y_list = zip(*batch)
    max_len = max(len(x) for x in x_list)
    
    padded_x = torch.full((len(batch), max_len), pad_idx, dtype=torch.long)
    padded_y = torch.full((len(batch), max_len), pad_idx, dtype=torch.long)
    
    for i in range(len(batch)):
        padded_x[i, :len(x_list[i])] = x_list[i]
        padded_y[i, :len(y_list[i])] = y_list[i]

    pad_mask = (padded_x != pad_idx).unsqueeze(1).unsqueeze(2) # Shape: (B, 1, 1, T)
    return padded_x, padded_y, pad_mask


class DatasetManager:
    """Analyzes datasets and provides summary token/character statistics."""
    def __init__(self, data_dir: str, tokenizer: BPETokenizer):
        self.data_dir = data_dir
        self.tokenizer = tokenizer

    def analyze(self) -> Dict[str, Any]:
        """Calculates total files, total characters, total tokens, vocab size, and average length."""
        files = glob.glob(os.path.join(self.data_dir, "*.txt"))
        raw_text = load_all_txt_files(self.data_dir)
        tokens = self.tokenizer.encode(raw_text)

        lines = raw_text.splitlines()
        avg_line_len = len(raw_text) / max(1, len(lines))

        return {
            "total_files": len(files),
            "total_characters": len(raw_text),
            "total_tokens": len(tokens),
            "vocab_size": len(self.tokenizer.encoder),
            "avg_line_length": round(avg_line_len, 2),
            "file_names": [os.path.basename(f) for f in files]
        }


def prepare_data(config, tokenizer: BPETokenizer):
    """Loads, tokenizes, splits data, and returns training/validation DataLoaders."""
    raw_text = load_all_txt_files(config.data_dir)
    token_ids = tokenizer.encode(raw_text)

    split_idx = int(0.9 * len(token_ids))
    train_tokens = token_ids[:split_idx]
    val_tokens = token_ids[split_idx:]

    train_ds = DynamicTextDataset(train_tokens, config.block_size)
    val_ds = DynamicTextDataset(val_tokens, config.block_size)

    pad_id = tokenizer.encoder.get(config.pad_token, 0)

    train_loader = DataLoader(
        train_ds, 
        batch_size=config.batch_size, 
        shuffle=True,
        collate_fn=lambda b: pad_collate_fn(b, pad_id)
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=config.batch_size, 
        shuffle=False,
        collate_fn=lambda b: pad_collate_fn(b, pad_id)
    )

    return train_loader, val_loader, len(token_ids)