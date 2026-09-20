"""Speaker embedding extraction using SpeechBrain ECAPA-TDNN."""

import os
import shutil
from typing import Optional
import numpy as np
import torch

# Safe symlink fallback for SpeechBrain on Windows (WinError 1314 fix)
_orig_symlink = getattr(os, "symlink", None)
def safe_symlink(src, dst, target_is_directory=False, *, dir_fd=None):
    try:
        if _orig_symlink:
            _orig_symlink(src, dst, target_is_directory=target_is_directory, dir_fd=dir_fd)
        else:
            shutil.copyfile(src, dst)
    except OSError as e:
        if getattr(e, 'winerror', None) == 1314 or getattr(e, 'errno', None) == 1314:
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if os.path.exists(dst):
                    os.remove(dst)
                shutil.copyfile(src, dst)
        else:
            raise

os.symlink = safe_symlink


class SpeakerEmbeddingExtractor:
    """Extracts 192-dimensional speaker embedding vectors from audio segments using ECAPA-TDNN."""

    def __init__(self, model_name: str = "speechbrain/spkrec-ecapa-voxceleb", device: Optional[str] = None):
        self.model_name = model_name
        
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "cpu"  # SpeechBrain pretrained runs best on CPU/CUDA
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        self._model = None

    def load_model(self) -> None:
        """Lazily load the speaker embedding model."""
        if self._model is None:
            try:
                from speechbrain.inference.speaker import EncoderClassifier
            except ImportError:
                from speechbrain.pretrained import EncoderClassifier

            run_device = "cpu" if self.device == "mps" else self.device
            self._model = EncoderClassifier.from_hparams(
                source=self.model_name,
                savedir=f"tmp_models/{os.path.basename(self.model_name)}",
                run_opts={"device": run_device}
            )

    def compute_embedding(self, audio_segment: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Compute 192-dim L2-normalized speaker embedding vector.
        
        Args:
            audio_segment: Mono float32 audio numpy array.
            sample_rate: Audio sampling rate (default 16000).
            
        Returns:
            L2-normalized 1D float32 numpy array of shape (192,).
        """
        self.load_model()

        if len(audio_segment) == 0:
            return np.zeros((192,), dtype=np.float32)

        # Minimum required length for ECAPA-TDNN conv filters (~3200 samples / 0.2s)
        min_len = 3200
        if len(audio_segment) < min_len:
            audio_segment = np.pad(audio_segment, (0, min_len - len(audio_segment)), mode='constant')

        signal_tensor = torch.from_numpy(audio_segment).float().unsqueeze(0)
        
        with torch.no_grad():
            embeddings = self._model.encode_batch(signal_tensor)
            
        emb_np = embeddings.squeeze().cpu().numpy().astype(np.float32)
        
        # Flatten if needed
        if emb_np.ndim > 1:
            emb_np = emb_np.squeeze()

        # L2 normalization
        norm = np.linalg.norm(emb_np)
        if norm > 0:
            emb_np = emb_np / norm

        return emb_np

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two speaker embeddings.
        
        Returns:
            Float scalar similarity score in range [-1.0, 1.0].
        """
        if len(emb1) == 0 or len(emb2) == 0:
            return 0.0
            
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        dot = float(np.dot(emb1, emb2) / (norm1 * norm2))
        return float(np.clip(dot, -1.0, 1.0))
