import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple

class SimpleVectorDB:
    """
    Pure PyTorch Educational In-Memory Vector Database.
    Calculates document embeddings and performs Cosine Similarity Search.
    
    Mathematical Cosine Similarity:
    Similarity(A, B) = (A . B) / (||A|| * ||B||)
    """
    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.documents: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.vectors: Optional[torch.Tensor] = None

    def _compute_simple_embedding(self, text: str) -> torch.Tensor:
        """
        Computes a deterministic pseudo-embedding vector for text chunk
        by combining character/ASCII frequencies and projecting to embedding_dim.
        """
        vec = torch.zeros(self.embedding_dim, dtype=torch.float32)
        for i, char in enumerate(text[:512]):
            idx = (ord(char) + i * 31) % self.embedding_dim
            vec[idx] += 1.0
        return F.normalize(vec, p=2, dim=0)

    def add_documents(self, docs: List[str], metadata: Optional[List[Dict[str, Any]]] = None):
        """Indexes documents into the vector database."""
        if not docs:
            return

        new_vectors = []
        for doc in docs:
            emb = self._compute_simple_embedding(doc)
            new_vectors.append(emb.unsqueeze(0))
            self.documents.append(doc)
            self.metadata.append(metadata.pop(0) if metadata else {})

        stacked_new = torch.cat(new_vectors, dim=0)
        if self.vectors is None:
            self.vectors = stacked_new
        else:
            self.vectors = torch.cat([self.vectors, stacked_new], dim=0)

    def search(self, query: str, top_k: int = 2) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Performs Cosine Similarity search over indexed vector space."""
        if self.vectors is None or len(self.documents) == 0:
            return []

        query_vec = self._compute_simple_embedding(query).unsqueeze(0) # (1, dim)
        
        # Cosine similarity matrix multiplication: (1, dim) @ (dim, N) -> (1, N)
        similarities = (query_vec @ self.vectors.T).squeeze(0) # (N,)

        scores, indices = torch.topk(similarities, min(top_k, len(self.documents)))

        results = []
        for score, idx in zip(scores.tolist(), indices.tolist()):
            results.append((self.documents[idx], round(score, 4), self.metadata[idx]))
            
        return results