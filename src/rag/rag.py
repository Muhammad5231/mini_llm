import os
import json
import csv
import re
from typing import List, Dict, Any
from src.vector_db.vector_db import SimpleVectorDB

class RAGPipeline:
    """
    Retrieval-Augmented Generation (RAG) Pipeline.
    Loads documents, splits into chunks, indexes into Vector DB, and builds grounding context.
    """
    def __init__(self, chunk_size: int = 120):
        self.chunk_size = chunk_size
        self.vector_db = SimpleVectorDB(embedding_dim=64)

    def _read_pdf_basic(self, filepath: str) -> str:
        """Extracts printable ASCII text streams from PDF files without third-party dependencies."""
        try:
            with open(filepath, "rb") as f:
                content = f.read()
            text_chunks = re.findall(rb'\(([^\)]+)\)', content)
            extracted = " ".join([c.decode("ascii", errors="ignore") for c in text_chunks])
            return extracted if extracted.strip() else "PDF text extraction failed."
        except Exception:
            return ""

    def load_document(self, filepath: str) -> str:
        """Reads file contents according to extension (TXT, JSON, CSV, MD, PDF)."""
        ext = os.path.splitext(filepath)[1].lower()
        if not os.path.exists(filepath):
            return ""

        if ext in [".txt", ".md"]:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

        elif ext == ".json":
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
                return json.dumps(data, indent=2)

        elif ext == ".csv":
            lines = []
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                for row in reader:
                    lines.append(" | ".join(row))
            return "\n".join(lines)

        elif ext == ".pdf":
            return self._read_pdf_basic(filepath)

        return ""

    def chunk_text(self, text: str) -> List[str]:
        """Splits long text into overlapping chunks of length chunk_size."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size // 2):
            chunk = " ".join(words[i : i + self.chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def index_directory(self, data_dir: str):
        """Indexes all supported text files in data_dir into Vector DB."""
        for root, _, files in os.walk(data_dir):
            for file in files:
                fpath = os.path.join(root, file)
                raw_text = self.load_document(fpath)
                if not raw_text.strip():
                    continue

                chunks = self.chunk_text(raw_text)
                metadata = [{"source": os.path.basename(fpath)} for _ in chunks]
                self.vector_db.add_documents(chunks, metadata)

    def retrieve_context(self, query: str, top_k: int = 2) -> str:
        """Retrieves relevant text chunks matching query and formats as context."""
        results = self.vector_db.search(query, top_k=top_k)
        if not results:
            return ""

        context_parts = []
        for i, (doc, score, meta) in enumerate(results):
            src = meta.get("source", "Knowledge Base")
            context_parts.append(f"[Source: {src} | Score: {score}]\n{doc}")

        return "\n\n".join(context_parts)