import json
import os
from collections import defaultdict
from typing import List, Dict, Tuple

class BPETokenizer:
    """
    Byte-Pair Encoding (BPE) Tokenizer built from scratch.
    
    Mathematical Mechanism:
    1. Initialize vocabulary with characters + special tokens.
    2. Count adjacent symbol pair frequencies: freq(u, v) = count of pair (u, v) across corpus.
    3. Iteratively merge the most frequent pair (u, v) into new token 'uv'.
    4. Repeat until vocabulary reaches target size or no pairs meet min_frequency.
    """
    def __init__(
        self, 
        vocab_size: int = 256, 
        min_frequency: int = 2,
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>"
    ):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        
        # Special Tokens
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.special_tokens = [pad_token, unk_token, bos_token, eos_token]

        # Maps token string -> integer ID
        self.encoder: Dict[str, int] = {}
        # Maps integer ID -> token string
        self.decoder: Dict[int, str] = {}
        # BPE merge rules ordered by priority: Tuple[str, str] -> merged string
        self.merges: List[Tuple[str, str]] = []

    def _get_stats(self, words: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], int]:
        """Calculates frequency of adjacent token pairs across corpus words."""
        pairs = defaultdict(int)
        for word, freq in words.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i+1])] += freq
        return pairs

    def _merge_word(self, word: Tuple[str, ...], pair: Tuple[str, str]) -> Tuple[str, ...]:
        """Replaces instances of `pair` inside a tuple word with the merged token string."""
        first, second = pair
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == first and word[i+1] == second:
                new_word.append(first + second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return tuple(new_word)

    def train(self, text: str):
        """Trains BPE merge rules on raw input text."""
        # 1. Initialize special tokens in vocabulary first
        vocab = list(self.special_tokens)

        # 2. Extract base character vocabulary from text
        raw_words = text.split()
        words_count: Dict[Tuple[str, ...], int] = defaultdict(int)
        
        for w in raw_words:
            # Represent each word as tuple of characters ending with word boundary marker '</w>'
            symbols = tuple(list(w) + ['</w>'])
            words_count[symbols] += 1
            for sym in symbols:
                if sym not in vocab:
                    vocab.append(sym)

        # 3. Iteratively find and merge most frequent symbol pairs
        num_merges = self.vocab_size - len(vocab)
        
        for _ in range(num_merges):
            pairs = self._get_stats(words_count)
            if not pairs:
                break
            
            # Select most frequent pair
            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < self.min_frequency:
                break  # Stop if pair frequency falls below minimum threshold
            
            # Merge best pair across all words
            words_count = {
                self._merge_word(word, best_pair): freq 
                for word, freq in words_count.items()
            }
            
            merged_token = best_pair[0] + best_pair[1]
            vocab.append(merged_token)
            self.merges.append(best_pair)

        # 4. Build final encoder and decoder dictionaries
        self.encoder = {token: idx for idx, token in enumerate(vocab)}
        self.decoder = {idx: token for idx, token in enumerate(vocab)}

    def encode(self, text: str) -> List[int]:
        """Encodes raw string into token IDs using learned BPE merge rules."""
        tokens = []
        words = text.split()

        for w in words:
            # Convert word to character tuple with end marker
            word_symbols = tuple(list(w) + ['</w>'])
            
            # Apply merges sequentially in order of priority
            for pair in self.merges:
                if len(word_symbols) <= 1:
                    break
                word_symbols = self._merge_word(word_symbols, pair)
            
            # Convert subword token strings to token IDs
            for sym in word_symbols:
                tokens.append(self.encoder.get(sym, self.encoder[self.unk_token]))

        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Decodes token IDs back to human-readable string."""
        raw_text = "".join([self.decoder.get(i, self.unk_token) for i in token_ids])
        # Replace word boundary indicator with space
        cleaned_text = raw_text.replace("</w>", " ").strip()
        return cleaned_text

    def save(self, filepath: str):
        """Saves vocabulary and merge rules to JSON."""
        data = {
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
            "encoder": self.encoder,
            "merges": self.merges,
            "special_tokens": self.special_tokens
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "BPETokenizer":
        """Loads BPE tokenizer state from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        tokenizer = cls(
            vocab_size=data["vocab_size"],
            min_frequency=data["min_frequency"]
        )
        tokenizer.encoder = data["encoder"]
        tokenizer.decoder = {int(v): k for k, v in data["encoder"].items()}
        tokenizer.merges = [tuple(p) for p in data["merges"]]
        return tokenizer