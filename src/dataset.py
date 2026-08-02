import os
import re
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from src.tokenizer import BPETokenizer

def clean_text(text: str) -> str:
    """
    Data Cleaning Pipeline:
    1. Replaces weird control characters with space while preserving newlines and punctuation.
    2. Normalizes multiple spaces/tabs into single space.
    """
    # Remove non-printable control characters except standard whitespace/newlines
    text = re.sub(r'[^\x20-\x7E\n\t]', '', text)
    # Collapse consecutive space/tab horizontal whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def load_all_txt_files(data_dir: str) -> str:
    """Scans directory for all .txt files, cleans and concatenates their contents."""
    files = glob.glob(os.path.join(data_dir, "*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in directory: {data_dir}")
    
    combined_text = []
    for fpath in sorted(files):
        with open(fpath, "r", encoding="utf-8") as f:
            raw = f.read()
            cleaned = clean_text(raw)
            combined_text.append(cleaned)
            
    return "\n\n".join(combined_text)


class DynamicTextDataset(Dataset):
    """
    PyTorch Dataset with chunking and dynamic sequence extraction.
    Creates sequence samples of length block_size for Next-Token Prediction.
    """
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
    """
    Dynamic Padding Collation function.
    Pads shorter sequences in a batch to match the maximum batch sequence length.
    Returns input batch, target batch, and padding mask tensor.
    """
    x_list, y_list = zip(*batch)
    
    # Calculate maximum actual sequence length in current batch
    max_len = max(len(x) for x in x_list)
    
    padded_x = torch.full((len(batch), max_len), pad_idx, dtype=torch.long)
    padded_y = torch.full((len(batch), max_len), pad_idx, dtype=torch.long)
    
    for i in range(len(batch)):
        padded_x[i, :len(x_list[i])] = x_list[i]
        padded_y[i, :len(y_list[i])] = y_list[i]

    # Attention Mask: 1 for real tokens, 0 for padded tokens
    pad_mask = (padded_x != pad_idx).unsqueeze(1).unsqueeze(2) # Shape: (B, 1, 1, T)

    return padded_x, padded_y, pad_mask


def prepare_data(config, tokenizer: BPETokenizer):
    """Loads text files, cleans, tokenizes, splits into train/val loaders."""
    raw_text = load_all_txt_files(config.data_dir)
    token_ids = tokenizer.encode(raw_text)

    # Train / Validation Split (90% Train, 10% Validation)
    split_idx = int(0.9 * len(token_ids))
    train_tokens = token_ids[:split_idx]
    val_tokens = token_ids[split_idx:]

    train_ds = DynamicTextDataset(train_tokens, config.block_size)
    val_ds = DynamicTextDataset(val_tokens, config.block_size)

    pad_id = tokenizer.encoder[config.pad_token]

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