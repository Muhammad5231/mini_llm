import os
import json
import csv
import logging
import time
from datetime import datetime
from typing import Dict, Any, List

class TrainingLogger:
    """Comprehensive Logging Manager for Mini LLM."""
    def __init__(self, log_dir: str, session_file: str):
        self.log_dir = log_dir
        self.session_file = session_file
        os.makedirs(log_dir, exist_ok=True)

        # Main Logger setup
        self.logger = logging.getLogger("MiniLLM")
        self.logger.setLevel(logging.INFO)
        
        # Avoid duplicate handlers on re-initialization
        if not self.logger.handlers:
            fh = logging.FileHandler(session_file, encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            self.logger.addHandler(fh)

    def log_info(self, message: str):
        self.logger.info(message)

    def log_error(self, message: str, exc: Optional[Exception] = None):
        if exc:
            self.logger.error(f"{message} | Exception: {str(exc)}")
        else:
            self.logger.error(message)

    def log_inference_perf(self, prompt_tokens: int, gen_tokens: int, duration_sec: float):
        tok_per_sec = gen_tokens / max(0.001, duration_sec)
        msg = f"[INFERENCE PERF] Prompt Toks: {prompt_tokens} | Gen Toks: {gen_tokens} | Time: {duration_sec:.3f}s | Speed: {tok_per_sec:.2f} tok/s"
        self.logger.info(msg)

    def save_history_json(self, history: List[Dict[str, Any]], json_path: str):
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    def save_history_csv(self, history: List[Dict[str, Any]], csv_path: str):
        if not history:
            return
        keys = history[0].keys()
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(history)