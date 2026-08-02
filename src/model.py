import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention.
    
    Equations:
    Q = X * W_Q,  K = X * W_K,  V = X * W_V
    Attention(Q, K, V) = softmax( (Q * K^T) / sqrt(d_k) + Mask ) * V
    """
    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        
        # Single projection linear layer for Q, K, V for speed
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

        # Lower triangular causal mask to prevent attention to future tokens
        # Shape: (1, 1, block_size, block_size)
        self.register_buffer(
            "bias", 
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size() # Batch size, Sequence length, Embedding dimension

        # Calculate Query, Key, Value vectors for all heads in batch
        # q, k, v shape: (B, T, C)
        q, k, v = self.c_attn(x).split(C, dim=2)

        # Reshape for multi-head attention: (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        # Scaled scores = (Q @ K^T) / sqrt(head_dim)
        # Shape: (B, n_head, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # Apply causal mask: set future positions to -infinity so softmax makes them 0
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        
        # Softmax along last dimension to get attention weights sum to 1
        att = F.softmax(att, dim=-1)
        
        # Weighted sum of values: (B, n_head, T, T) @ (B, n_head, T, head_dim) -> (B, n_head, T, head_dim)
        y = att @ v 
        
        # Re-assemble all head outputs side-by-side
        # Shape: (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        return self.c_proj(y)


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (MLP).
    
    Equation:
    FFN(x) = GELU(x * W_1 + b_1) * W_2 + b_2
    Standard expansion factor is 4x hidden dimension.
    """
    def __init__(self, n_embd: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Standard Decoder Transformer Block.
    Uses Pre-LayerNormalization architecture:
    x = x + SelfAttention(LayerNorm(x))
    x = x + FeedForward(LayerNorm(x))
    """
    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, block_size)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = FeedForward(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual Connections (x + ...)
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MiniLLM(nn.Module):
    """
    Complete Causal Decoder Transformer Language Model.
    
    Total parameters formula roughly:
    Params = Vocab_Size * Embed_Size + Layers * (4 * Embed_Size^2 + 8 * Embed_Size^2)
    """
    def __init__(self, vocab_size: int, n_embd: int = 64, n_layer: int = 2, n_head: int = 2, block_size: int = 64):
        super().__init__()
        self.block_size = block_size

        # Token Embedding Table: Maps token index -> vector of size n_embd
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        
        # Positional Embedding Table: Maps position index (0 to block_size-1) -> vector
        self.pos_emb = nn.Embedding(block_size, n_embd)

        # Stack of Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, block_size) for _ in range(n_layer)
        ])

        # Final Layer Normalization
        self.ln_f = nn.LayerNorm(n_embd)

        # Language Model Head (maps embedding vectors to vocabulary logits)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying (share weights between token embedding and final projection)
        self.tok_emb.weight = self.lm_head.weight

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = idx.size()
        assert T <= self.block_size, f"Cannot forward sequence of length {T}, block size is {self.block_size}"

        # Get positions array [0, 1, ..., T-1]
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        # Combine Token Embeddings and Positional Embeddings
        # Token Emb: (B, T, n_embd), Pos Emb: (T, n_embd) -> Broadcasted addition
        x = self.tok_emb(idx) + self.pos_emb(pos)

        # Forward pass through all Transformer Blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x) # (B, T, n_embd)

        if targets is not None:
            # Compute logits for all vocabulary characters
            logits = self.lm_head(x) # (B, T, vocab_size)
            
            # Reshape tensors for PyTorch CrossEntropyLoss calculation
            # Loss = -log( softmax(logits_true_class) )
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            # Inference optimization: focus only on the last time step logits
            logits = self.lm_head(x[:, -1:, :]) # (B, 1, vocab_size)
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Autoregressive generation loop.
        Appends predictions to context step by step.
        """
        for _ in range(max_new_tokens):
            # Crop current sequence to max context window size if it exceeds block_size
            idx_cond = idx[:, -self.block_size:]
            
            # Get predictions
            logits, _ = self(idx_cond)
            
            # Select last token logits and convert to probabilities using Softmax
            logits = logits[:, -1, :] # (B, vocab_size)
            probs = F.softmax(logits, dim=-1) # (B, vocab_size)
            
            # Sample next token from categorical distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            # Append sampled token to current sequence
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx