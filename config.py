import os
import torch
from dataclasses import dataclass

@dataclass
class Config:
    """
    Central Configuration class for Mini LLM (Version 3).
    Centralizes Decoder-Only Transformer parameters, training controls, and paths.
    """
    # System & Reproducibility
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = torch.cuda.is_available()  # Automatic Mixed Precision for GPU if available

    # Tokenizer Settings
    vocab_size: int = 256          # Target BPE vocabulary size
    min_frequency: int = 2         # Minimum pair frequency for BPE merges
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"

    # Decoder-Only Architecture (Version 3)
    n_embd: int = 64               # Hidden dimension (d_model)
    n_layer: int = 2               # Transformer layers
    n_head: int = 2                # Multi-head attention count
    block_size: int = 64           # Context window length (T)
    dropout: float = 0.1           # Dropout rate
    norm_eps: float = 1e-5         # RMSNorm epsilon for numerical stability
    activation_type: str = "swiglu" # Options: "swiglu", "gelu", "relu"
    use_rope: bool = True          # Use Rotary Position Embeddings
    use_rmsnorm: bool = True       # Use RMSNorm instead of LayerNorm

    # Training & Optimization
    batch_size: int = 8            # Physical batch size on CPU
    grad_accum_steps: int = 2      # Effective batch size = batch_size * grad_accum_steps = 16
    epochs: int = 30
    learning_rate: float = 1e-3
    min_lr: float = 1e-5           # Minimum LR for Cosine Decay
    warmup_epochs: int = 3         # Epochs for linear warmup
    max_grad_norm: float = 1.0     # Gradient clipping threshold
    early_stopping_patience: int = 5 # Epochs to wait before early stopping
    save_every_n_epochs: int = 5

    # Inference & Generation
    temperature: float = 0.8
    top_k: int = 20
    top_p: float = 0.9
    repetition_penalty: float = 1.2
    max_new_tokens: int = 80
    use_kv_cache: bool = True      # Fast inference with Key-Value Cache

    # Paths
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"
    export_dir: str = "exports"
    log_dir: str = "logs"
    plot_dir: str = "plots"
    tokenizer_path: str = os.path.join("checkpoints", "bpe_tokenizer.json")
    best_model_path: str = os.path.join("checkpoints", "mini_llm_best.pt")
    latest_model_path: str = os.path.join("checkpoints", "mini_llm_latest.pt")
    loss_plot_path: str = os.path.join("plots", "loss_curve.png")
    history_json_path: str = os.path.join("logs", "train_history.json")
    history_csv_path: str = os.path.join("logs", "train_history.csv")
    session_log_path: str = os.path.join("logs", "session.log")

    def __post_init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.export_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)