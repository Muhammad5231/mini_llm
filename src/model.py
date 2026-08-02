import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed Sinusoidal Positional Encoding.
    
    Equations:
    PE(pos, 2i)   = sin(pos / (10000^(2i / d_model)))
    PE(pos, 2i+1) = cos(pos / (10000^(2i / d_model)))
    """
    def __init__(self, n_embd: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, n_embd)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_embd, 2).float() * (-math.log(10000.0) / n_embd))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Shape: (1, max_len, n_embd)
        
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add positional encoding vectors up to sequence length T
        return x + self.pe[:, :x.size(1), :]


class DynamicMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention supporting both Causal Masking and Dynamic Padding Masking.
    """
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal triangular mask (lower triangular = 1, upper triangular = 0)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor = None) -> torch.Tensor:
        B, T, C = x.size()

        # Query, Key, Value Projections
        q, k, v = self.c_attn(x).split(C, dim=2)

        # Reshape to (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention Scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim)) # (B, n_head, T, T)

        # Combine Causal Mask with Padding Mask
        causal = self.causal_mask[:, :, :T, :T]
        att = att.masked_fill(causal == 0, float('-inf'))
        
        if pad_mask is not None:
            # pad_mask is 0 for padding tokens
            att = att.masked_fill(pad_mask == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Compute output vectors
        y = att @ v # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_dropout(self.c_proj(y))


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Neural Network with GELU activation."""
    def __init__(self, n_embd: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Transformer Decoder Block with Residual Connections & Layer Normalization."""
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = DynamicMultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = FeedForward(n_embd, dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), pad_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class MiniLLM(nn.Module):
    """Complete Educational Mini LLM Version 2."""
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = SinusoidalPositionalEncoding(config.n_embd, config.block_size)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(config.n_embd, config.n_head, config.block_size, config.dropout)
            for _ in range(config.n_layer)
        ])

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight Tying
        self.tok_emb.weight = self.lm_head.weight

        # Parameter initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None, pad_mask: torch.Tensor = None):
        x = self.tok_emb(idx)
        x = self.pos_emb(x)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x, pad_mask)

        x = self.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        else:
            logits = self.lm_head(x[:, -1:, :])
            loss = None

        return logits, loss

    def get_model_summary(self) -> dict:
        """Calculates total parameters, trainable parameters, and model size in MB."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # 4 bytes per float32 parameter
        size_bytes = total_params * 4
        size_mb = size_bytes / (1024 * 1024)

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": round(size_mb, 4),
            "vocab_size": self.config.vocab_size
        }