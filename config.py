import os
import json
import torch
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class Config:
    """
    Central Configuration class for Mini LLM (Version 4 Extended).
    Supports backward compatibility, JSON/ENV overrides, RAG, Memory, Vector DB, Tools & REST API.
    """
    # System & Reproducibility
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = torch.cuda.is_available()

    # Tokenizer Settings
    vocab_size: int = 256
    min_frequency: int = 1
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"

    # Decoder-Only Architecture
    n_embd: int = 64
    n_layer: int = 2
    n_head: int = 2
    block_size: int = 64
    dropout: float = 0.1
    norm_eps: float = 1e-5
    activation_type: str = "swiglu"
    use_rope: bool = True
    use_rmsnorm: bool = True

    # Training & Optimization
    batch_size: int = 8
    grad_accum_steps: int = 2
    epochs: int = 30
    learning_rate: float = 1e-3
    min_lr: float = 1e-5
    warmup_epochs: int = 3
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 10
    save_every_n_epochs: int = 5

    # Inference & Sampling
    temperature: float = 0.1
    top_k: int = 5
    top_p: float = 0.8
    repetition_penalty: float = 1.2
    max_new_tokens: int = 80
    use_kv_cache: bool = True

    # Version 4 Features Settings
    enable_memory: bool = True
    memory_history_limit: int = 5
    enable_rag: bool = True
    rag_top_k: int = 2
    rag_chunk_size: int = 120
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    enable_tools: bool = True

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
        self._apply_env_overrides()

    def _apply_env_overrides(self):
        """Overrides parameters from environment variables (e.g. MINILLM_LEARNING_RATE=0.0005)."""
        for key in self.__dataclass_fields__:
            env_var = f"MINILLM_{key.upper()}"
            if env_var in os.environ:
                val = os.environ[env_var]
                field_type = type(getattr(self, key))
                if field_type == bool:
                    setattr(self, key, val.lower() in ("true", "1", "yes"))
                else:
                    setattr(self, key, field_type(val))

    def load_from_json(self, json_path: str):
        """Loads configuration overrides from JSON file."""
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.items():
                if hasattr(self, k):
                    setattr(self, k, v)

    def save_to_json(self, json_path: str):
        """Exports current configuration to JSON file."""
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, indent=2)