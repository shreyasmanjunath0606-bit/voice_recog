"""Speaker embedding extraction (e.g. ECAPA-TDNN / Resemblyzer / PyAnnote)."""

from typing import Optional
import numpy as np


class SpeakerEmbeddingExtractor:
    """Extracts speaker embedding vectors from audio segments."""

    def __init__(self, model_name: str = "speechbrain/spkrec-ecapa-voxceleb", device: str = "mps"):
        self.model_name = model_name
        self.device = device
        self._model = None

    def load_model(self) -> None:
        """Lazily load the speaker embedding model."""
        pass

    def compute_embedding(self, audio_segment: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Compute 192/512-dim normalized speaker embedding vector."""
        return np.zeros((192,), dtype=np.float32)

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two speaker embeddings."""
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))
