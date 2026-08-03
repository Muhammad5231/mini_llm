import os
import json
from typing import List, Dict, Any, Optional

class ConversationMemory:
    """
    Manages short-term conversation context and persistent memory storage.
    Allows LLM to remember past user interactions across restarts.
    """
    def __init__(self, storage_path: str = os.path.join("logs", "conversation_memory.json")):
        self.storage_path = storage_path
        self.history: List[Dict[str, str]] = []
        self.load_memory()

    def add_turn(self, user_text: str, assistant_text: str):
        """Appends user query and assistant answer to memory."""
        self.history.append({
            "user": user_text,
            "assistant": assistant_text
        })
        self.save_memory()

    def get_recent_context(self, max_turns: int = 3) -> str:
        """Formats the last N conversation turns into context prompt format."""
        turns = self.history[-max_turns:]
        formatted = []
        for turn in turns:
            formatted.append(f"User: {turn['user']}\nAssistant: {turn['assistant']}")
        return "\n\n".join(formatted)

    def search_memory(self, query: str, limit: int = 2) -> List[Dict[str, str]]:
        """Simple keyword-based memory retrieval."""
        keywords = query.lower().split()
        results = []
        for turn in reversed(self.history):
            text = f"{turn['user']} {turn['assistant']}".lower()
            if any(kw in text for kw in keywords):
                results.append(turn)
                if len(results) >= limit:
                    break
        return results

    def clear(self):
        """Clears memory history."""
        self.history = []
        self.save_memory()

    def save_memory(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

    def load_memory(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    self.history = json.load(f)
            except Exception:
                self.history = []