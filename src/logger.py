import os
import json
import csv
import logging
from datetime import datetime

class TrainingLogger:
    """Handles logging to console, session log files, JSON, and CSV format."""
    def __init__(self, log_dir: str, session_file: str):
        self.log_dir = log_dir
        self.session_file = session_file
        os.makedirs(log_dir, exist_ok=True)

        # Configure Python standard logging
        logging.basicConfig(
            filename=session_file,
            filemode="a",
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO
        )
        self.logger = logging.getLogger("MiniLLM")

    def log_info(self, message: str):
        """Logs message to file and console."""
        self.logger.info(message)

    def save_history_json(self, history: list[dict], json_path: str):
        """Saves epoch training records to JSON."""
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    def save_history_csv(self, history: list[dict], csv_path: str):
        """Saves epoch training records to CSV."""
        if not history:
            return
        keys = history[0].keys()
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(history)