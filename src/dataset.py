import torch
from torch.utils.data import Dataset, DataLoader

class CharTokenizer:
    """
    Character-level Tokenizer.
    Maps characters to integers (encoding) and integers to characters (decoding).
    """
    def __init__(self, text: str):
        # Extract unique characters and sort them to form the vocabulary
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        
        # String-to-Integer mapping
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        # Integer-to-String mapping
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s: str) -> list[int]:
        """Convert string to list of character token IDs."""
        return [self.stoi[c] for c in s]

    def decode(self, l: list[int]) -> str:
        """Convert list of character token IDs back to string."""
        return ''.join([self.itos[i] for i in l])


class TextDataset(Dataset):
    """
    PyTorch Dataset for Next-Token Prediction.
    
    Given a sequence of tokens of length block_size:
    Input (x)  : text[i : i + block_size]
    Target (y) : text[i + 1 : i + block_size + 1]
    
    Mathematical relation:
    y[t] = x[t + 1]
    The model predicts token at index (t + 1) given sequence up to index t.
    """
    def __init__(self, data_tensor: torch.Tensor, block_size: int):
        self.data = data_tensor
        self.block_size = block_size

    def __len__(self) -> int:
        # Total valid starting indices
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Chunk of size block_size for input x
        x = self.data[idx : idx + self.block_size]
        # Shifted chunk of size block_size for target y
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


def prepare_dataloaders(
    text: str, 
    block_size: int = 64, 
    batch_size: int = 16, 
    split_ratio: float = 0.9
):
    """
    Processes raw text, instantiates Tokenizer, splits data into train/val,
    and returns PyTorch DataLoaders.
    """
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    
    # Train/Val split
    n = int(split_ratio * len(data))
    train_data = data[:n]
    val_data = data[n:]

    train_dataset = TextDataset(train_data, block_size)
    val_dataset = TextDataset(val_data, block_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, tokenizer