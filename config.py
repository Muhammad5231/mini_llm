import os
import torch
from dataclasses import dataclass

@dataclass
class Config:
    """
    Central Configuration class for Mini LLM (Version 2).
    Centralizes model dimensions, training parameters, sampling options, and paths.
    """
    # System & Reproducibility
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Tokenizer Settings
    vocab_size: int = 256          # Target BPE vocabulary size for small corpus
    min_frequency: int = 2         # Minimum frequency for BPE pair merging
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"

    # Model Architecture
    n_embd: int = 64               # Embedding dimension size (d_model)
    n_layer: int = 2               # Number of Transformer blocks
    n_head: int = 2                # Number of Attention heads
    block_size: int = 64           # Maximum context length (T)
    dropout: float = 0.1           # Dropout rate for regularization

    # Training Hyperparameters
    batch_size: int = 16
    epochs: int = 30
    learning_rate: float = 1e-3
    min_lr: float = 1e-5           # Minimum learning rate for cosine scheduler
    warmup_epochs: int = 3         # Epochs for linear warmup
    max_grad_norm: float = 1.0     # Maximum gradient norm for clipping
    save_every_n_epochs: int = 5   # Checkpoint saving interval

    # Inference & Generation
    temperature: float = 0.8       # Temperature for logit scaling (T > 0)
    top_k: int = 20                # Top-K sampling cutoff
    top_p: float = 0.9             # Top-P (Nucleus) cumulative probability cutoff
    repetition_penalty: float = 1.2 # Penalty factor for repeated tokens (> 1.0)
    max_new_tokens: int = 80       # Default maximum tokens to generate

    # Paths
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"
    plot_dir: str = "plots"
    tokenizer_path: str = os.path.join("checkpoints", "bpe_tokenizer.json")
    best_model_path: str = os.path.join("checkpoints", "mini_llm_best.pt")
    latest_model_path: str = os.path.join("checkpoints", "mini_llm_latest.pt")
    loss_plot_path: str = os.path.join("plots", "loss_curve.png")

    def __post_init__(self):
        # Create necessary directories automatically
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)