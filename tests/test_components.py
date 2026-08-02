import unittest
import torch
from config import Config
from src.model import RMSNorm, SwiGLUFeedForward, RotaryPositionalEmbedding, MultiHeadCausalAttention
from src.tokenizer import BPETokenizer
from src.dataset import DynamicTextDataset

class TestTransformerComponents(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.config.n_embd = 64
        self.config.n_head = 2
        self.config.block_size = 32

    def test_rmsnorm_shape(self):
        norm = RMSNorm(self.config.n_embd)
        x = torch.randn(4, 16, self.config.n_embd)
        out = norm(x)
        self.assertEqual(out.shape, x.shape)

    def test_swiglu_shape(self):
        mlp = SwiGLUFeedForward(self.config.n_embd)
        x = torch.randn(4, 16, self.config.n_embd)
        out = mlp(x)
        self.assertEqual(out.shape, x.shape)

    def test_rope_rotation(self):
        rope = RotaryPositionalEmbedding(dim=32, max_len=64)
        q = torch.randn(2, 2, 16, 32)
        cos, sin = rope(q, seq_len=16)
        rotated_q = rope.apply_rope(q, cos, sin)
        self.assertEqual(rotated_q.shape, q.shape)

    def test_attention_shape_and_cache(self):
        attn = MultiHeadCausalAttention(self.config)
        x = torch.randn(2, 16, self.config.n_embd)
        out = attn(x)
        self.assertEqual(out.shape, x.shape)

    def test_bpe_tokenizer_encode_decode(self):
        tokenizer = BPETokenizer(vocab_size=100, min_frequency=1)
        text = "hello world artificial intelligence"
        tokenizer.train(text)
        encoded = tokenizer.encode("hello world")
        decoded = tokenizer.decode(encoded)
        self.assertIsInstance(encoded, list)
        self.assertTrue(len(decoded) > 0)

    def test_dataset_shape(self):
        tokens = list(range(100))
        ds = DynamicTextDataset(tokens, block_size=16)
        x, y = ds[0]
        self.assertEqual(len(x), 16)
        self.assertEqual(len(y), 16)

if __name__ == "__main__":
    unittest.main()