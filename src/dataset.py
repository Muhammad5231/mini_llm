import os
import re
import glob
import hashlib
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Any, Optional
from src.tokenizer import BPETokenizer

def scan_txt_files_recursively(data_dir: str) -> List[str]:
    """
    CRITICAL BUG FIX:
    Recursively scans directory and ALL subdirectories at unlimited depth for .txt files.
    Examples matched:
    - data/input.txt
    - data/english/Wikipedia/computer.txt
    - data/hinglish/Conversation.txt
    """
    txt_files = []
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(".txt"):
                txt_files.append(os.path.join(root, file))

    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in '{data_dir}' or any of its subdirectories.")
    
    return sorted(txt_files)


def read_file_with_fallback_encoding(filepath: str) -> str:
    """Reads file attempting UTF-8 first, falling back to Latin-1 and CP1252 if needed."""
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    for enc in encodings:
        try:
            with open(filepath, "r", encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    # Force read ignoring errors if all fail
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def clean_text(text: str) -> str:
    """Data Cleaning: Preserves standard characters, punctuation, whitespace, and newlines."""
    text = re.sub(r'[^\x20-\x7E\n\tĠ]', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def load_all_txt_files(data_dir: str) -> str:
    """Scans all nested subfolders, cleans text, removes duplicate documents, and concatenates."""
    files = scan_txt_files_recursively(data_dir)
    seen_hashes = set()
    combined = []

    for fpath in files:
        raw = read_file_with_fallback_encoding(fpath)
        cleaned = clean_text(raw)
        if not cleaned:
            continue

        # Hash-based duplicate removal
        doc_hash = hashlib.md5(cleaned.encode("utf-8")).hexdigest()
        if doc_hash not in seen_hashes:
            seen_hashes.add(doc_hash)
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
    """Dataset Manager: Deep recursive statistics, language detection, and duplicate audit."""
    def __init__(self, data_dir: str, tokenizer: BPETokenizer):
        self.data_dir = data_dir
        self.tokenizer = tokenizer

    def analyze(self) -> Dict[str, Any]:
        """Calculates file count, nested directory stats, token stats, and duplicate metrics."""
        files = scan_txt_files_recursively(self.data_dir)
        raw_text = load_all_txt_files(self.data_dir)
        tokens = self.tokenizer.encode(raw_text)

        lines = [line for line in raw_text.splitlines() if line.strip()]
        
        # Simple Language Heuristic
        hinglish_keywords = ["hai", "kya", "aur", "kaise", "raha", "ho", "sakte"]
        hinglish_count = sum(raw_text.lower().count(kw) for kw in hinglish_keywords)
        detected_lang = "Hinglish / English Mixed" if hinglish_count > 5 else "English"

        return {
            "total_files_discovered": len(files),
            "detected_primary_language": detected_lang,
            "total_characters": len(raw_text),
            "total_tokens": len(tokens),
            "vocab_size": len(self.tokenizer.encoder),
            "total_non_empty_lines": len(lines),
            "avg_line_length": round(len(raw_text) / max(1, len(lines)), 2),
            "file_paths": [os.path.relpath(f, self.data_dir) for f in files]
        }


def prepare_data(config, tokenizer: BPETokenizer):
    """Loads, tokenizes, splits data recursively, and returns DataLoaders."""
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