import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm) implemented from scratch.
    
    Mathematical Equation:
    y = (x / sqrt( mean(x^2) + eps )) * gamma
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.gamma


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE) implemented from scratch.
    Rotates Query and Key vectors in 2D vector pairs by angular frequencies.
    """
    def __init__(self, dim: int, max_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        
        # persistent=False keeps buffers out of saved state_dict checkpoints
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(max_len, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, inv_freq) # Shape: (max_len, dim // 2)
        emb = torch.cat((freqs, freqs), dim=-1)           # Shape: (max_len, dim)
        
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(0), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : self.dim // 2]
        x2 = x[..., self.dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, seq_len: int, start_pos: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos_cached[:, :, start_pos : start_pos + seq_len, :]
        sin = self.sin_cached[:, :, start_pos : start_pos + seq_len, :]
        return cos, sin

    def apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        return (x * cos) + (self._rotate_half(x) * sin)


class SwiGLUFeedForward(nn.Module):
    """Swish-Gated Linear Unit (SwiGLU) Feed-Forward Network."""
    def __init__(self, n_embd: int, dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(2 * (4 * n_embd) / 3)
        self.w_gate = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w_up = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class MultiHeadCausalAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention with RoPE & Key-Value (KV) Cache support.
    """
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.block_size = config.block_size

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # persistent=False keeps non-trainable causal mask out of saved state_dict
        max_mask_len = 2048
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_mask_len, max_mask_len)).view(1, 1, max_mask_len, max_mask_len),
            persistent=False
        )

        self.cache_k: Optional[torch.Tensor] = None
        self.cache_v: Optional[torch.Tensor] = None

    def reset_cache(self):
        self.cache_k = None
        self.cache_v = None

    def forward(
        self, 
        x: torch.Tensor, 
        rope: Optional[RotaryPositionalEmbedding] = None,
        use_cache: bool = False,
        start_pos: int = 0,
        pad_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if rope is not None:
            cos, sin = rope(q, seq_len=T, start_pos=start_pos)
            q = rope.apply_rope(q, cos, sin)
            k = rope.apply_rope(k, cos, sin)

        if use_cache:
            if self.cache_k is None or start_pos == 0:
                self.cache_k = k
                self.cache_v = v
            else:
                self.cache_k = torch.cat([self.cache_k, k], dim=2)
                self.cache_v = torch.cat([self.cache_v, v], dim=2)
            k = self.cache_k
            v = self.cache_v

        total_T = k.size(2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        if not use_cache:
            causal = self.causal_mask[:, :, :T, :total_T]
            att = att.masked_fill(causal == 0, float('-inf'))

        if pad_mask is not None and not use_cache:
            att = att.masked_fill(pad_mask == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_dropout(self.c_proj(y))


class DecoderBlock(nn.Module):
    """Transformer Decoder Layer with RMSNorm, Self-Attention & SwiGLU MLP."""
    def __init__(self, config):
        super().__init__()
        self.norm_1 = RMSNorm(config.n_embd, eps=config.norm_eps) if config.use_rmsnorm else nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadCausalAttention(config)
        self.norm_2 = RMSNorm(config.n_embd, eps=config.norm_eps) if config.use_rmsnorm else nn.LayerNorm(config.n_embd)
        
        if config.activation_type == "swiglu":
            self.mlp = SwiGLUFeedForward(config.n_embd, config.dropout)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(config.n_embd, 4 * config.n_embd),
                nn.GELU() if config.activation_type == "gelu" else nn.ReLU(),
                nn.Linear(4 * config.n_embd, config.n_embd),
                nn.Dropout(config.dropout)
            )

    def forward(
        self, 
        x: torch.Tensor, 
        rope: Optional[RotaryPositionalEmbedding] = None,
        use_cache: bool = False,
        start_pos: int = 0,
        pad_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + self.attn(self.norm_1(x), rope=rope, use_cache=use_cache, start_pos=start_pos, pad_mask=pad_mask)
        x = x + self.mlp(self.norm_2(x))
        return x


class MiniLLM(nn.Module):
    """Complete Decoder-Only GPT Language Model (Version 3)."""
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.rope = RotaryPositionalEmbedding(config.n_embd // config.n_head, max_len=2048) if config.use_rope else None
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)])
        
        self.norm_f = RMSNorm(config.n_embd, eps=config.norm_eps) if config.use_rmsnorm else nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.tok_emb.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def reset_kv_cache(self):
        for block in self.blocks:
            block.attn.reset_cache()

    def forward(
        self, 
        idx: torch.Tensor, 
        targets: Optional[torch.Tensor] = None, 
        pad_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        start_pos: int = 0
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.tok_emb(idx)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x, rope=self.rope, use_cache=use_cache, start_pos=start_pos, pad_mask=pad_mask)

        x = self.norm_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        else:
            logits = self.lm_head(x[:, -1:, :])
            loss = None

        return logits, loss

    def get_model_summary(self) -> dict:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        size_mb = (total_params * 4) / (1024 * 1024)
        estimated_ram_mb = size_mb * 4

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": round(size_mb, 4),
            "estimated_ram_mb": round(estimated_ram_mb, 2),
            "n_layer": self.config.n_layer,
            "n_embd": self.config.n_embd,
            "n_head": self.config.n_head,
            "vocab_size": self.config.vocab_size,
            "architecture": "Decoder-Only Transformer (RoPE + RMSNorm + SwiGLU)"
        }