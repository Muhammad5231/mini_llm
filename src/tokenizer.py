import json
import re
from collections import defaultdict
from typing import List, Dict, Tuple

class BPETokenizer:
    """
    Byte-Pair Encoding (BPE) Tokenizer with Space-Prefix ('Ġ') and Newline ('\n') Preservation.
    Ensures generated text retains spaces, formatting, and line breaks.
    """
    def __init__(
        self, 
        vocab_size: int = 256, 
        min_frequency: int = 1,
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>"
    ):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.special_tokens = [pad_token, unk_token, bos_token, eos_token]

        self.encoder: Dict[str, int] = {}
        self.decoder: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []

    def _tokenize_raw_words(self, text: str) -> List[str]:
        """
        Splits raw text while explicitly capturing spaces (as 'Ġ') and newlines ('\n').
        """
        tokens = re.findall(r'\n|\s*\S+', text)
        processed = []
        for t in tokens:
            if t == '\n':
                processed.append('\n')
            else:
                leading_spaces = len(t) - len(t.lstrip(' '))
                word = 'Ġ' * leading_spaces + t.lstrip(' ')
                if word:
                    processed.append(word)
        return processed

    def _get_stats(self, words: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], int]:
        pairs = defaultdict(int)
        for word, freq in words.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i+1])] += freq
        return pairs

    def _merge_word(self, word: Tuple[str, ...], pair: Tuple[str, str]) -> Tuple[str, ...]:
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
        """Trains BPE vocabulary preserving spaces and newlines."""
        vocab = list(self.special_tokens)
        words = self._tokenize_raw_words(text)

        words_count: Dict[Tuple[str, ...], int] = defaultdict(int)
        for w in words:
            symbols = tuple(list(w))
            words_count[symbols] += 1
            for sym in symbols:
                if sym not in vocab:
                    vocab.append(sym)

        num_merges = self.vocab_size - len(vocab)
        for _ in range(max(0, num_merges)):
            pairs = self._get_stats(words_count)
            if not pairs:
                break
            
            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < self.min_frequency:
                break

            words_count = {
                self._merge_word(word, best_pair): freq 
                for word, freq in words_count.items()
            }
            
            merged_token = best_pair[0] + best_pair[1]
            vocab.append(merged_token)
            self.merges.append(best_pair)

        self.encoder = {token: idx for idx, token in enumerate(vocab)}
        self.decoder = {idx: token for idx, token in enumerate(vocab)}

    def encode(self, text: str) -> List[int]:
        """Encodes string to token IDs."""
        tokens = []
        raw_words = self._tokenize_raw_words(text)
        for w in raw_words:
            word_symbols = tuple(list(w))
            for pair in self.merges:
                if len(word_symbols) <= 1:
                    break
                word_symbols = self._merge_word(word_symbols, pair)
            for sym in word_symbols:
                tokens.append(self.encoder.get(sym, self.encoder[self.unk_token]))
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Decodes token IDs back to human-readable string with restored spaces."""
        raw_text = "".join([self.decoder.get(i, "") for i in token_ids])
        return raw_text.replace("Ġ", " ")

    def save(self, filepath: str):
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