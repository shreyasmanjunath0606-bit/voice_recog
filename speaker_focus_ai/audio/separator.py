"""Speech source separation and target-speaker extraction using SpeechBrain SepFormer."""

import os
import shutil
from typing import List, Optional, Tuple
import numpy as np
import torch

from speaker_focus_ai.audio.speaker_embedding import SpeakerEmbeddingExtractor

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


class AudioSeparator:
    """Separates target speaker voice from mixed multi-speaker audio."""

    def __init__(
        self,
        model_name: str = "speechbrain/sepformer-wham",
        device: Optional[str] = None,
        embedding_extractor: Optional[SpeakerEmbeddingExtractor] = None
    ):
        self.model_name = model_name
        
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.embedding_extractor = embedding_extractor or SpeakerEmbeddingExtractor(device=self.device)
        self._model = None

    def load_model(self) -> None:
        """Lazily load source separation model."""
        if self._model is None:
            try:
                from speechbrain.inference.separation import SepformerSeparation
            except ImportError:
                from speechbrain.pretrained import SepformerSeparation

            run_device = "cpu" if self.device == "mps" else self.device
            self._model = SepformerSeparation.from_hparams(
                source=self.model_name,
                savedir=f"tmp_models/{os.path.basename(self.model_name)}",
                run_opts={"device": run_device}
            )

    def blind_separate(
        self, mixed_waveform: np.ndarray, sample_rate: int = 16000, num_sources: int = 2
    ) -> List[np.ndarray]:
        """Blind separation of mixed audio into N separate source streams.
        
        Args:
            mixed_waveform: Float32 numpy array of mixed audio.
            sample_rate: Audio sampling rate (default 16000).
            num_sources: Expected number of sources.
            
        Returns:
            List of 1D float32 numpy arrays for each separated speaker stream.
        """
        self.load_model()

        if len(mixed_waveform) == 0:
            return [np.array([], dtype=np.float32) for _ in range(num_sources)]

        # Minimum required length for convolution frames
        min_len = 3200
        original_len = len(mixed_waveform)
        if original_len < min_len:
            mixed_waveform = np.pad(mixed_waveform, (0, min_len - original_len), mode='constant')

        signal_tensor = torch.from_numpy(mixed_waveform).float().unsqueeze(0)

        with torch.no_grad():
            est_sources = self._model.separate_batch(signal_tensor)

        # Output shape: (1, samples, num_sources) or (1, num_sources, samples)
        est_sources_np = est_sources.squeeze(0).cpu().numpy().astype(np.float32)

        streams = []
        if est_sources_np.ndim == 2:
            # Check dimensions (samples vs sources)
            if est_sources_np.shape[0] < est_sources_np.shape[1]:
                # Shape is (num_sources, samples)
                for i in range(est_sources_np.shape[0]):
                    streams.append(est_sources_np[i, :original_len])
            else:
                # Shape is (samples, num_sources)
                for i in range(est_sources_np.shape[1]):
                    streams.append(est_sources_np[:original_len, i])
        else:
            streams = [mixed_waveform[:original_len]]

        return streams

    def separate_target(
        self,
        mixed_waveform: np.ndarray,
        sample_rate: int = 16000,
        voiceprint: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """Extract the target speaker waveform, conditioned on voiceprint if available.
        
        Args:
            mixed_waveform: Float32 audio array.
            sample_rate: Audio sampling rate.
            voiceprint: Optional 192-dim target speaker voiceprint vector.
            
        Returns:
            Tuple of (target_waveform, separation_confidence).
        """
        streams = self.blind_separate(mixed_waveform, sample_rate=sample_rate)
        if not streams:
            return mixed_waveform, 0.0

        if voiceprint is None or len(voiceprint) == 0:
            return streams[0], 0.70

        # Match separated streams against voiceprint embedding
        best_stream_idx = 0
        best_similarity = -1.0

        for idx, stream in enumerate(streams):
            if len(stream) == 0:
                continue
            stream_emb = self.embedding_extractor.compute_embedding(stream, sample_rate=sample_rate)
            sim = self.embedding_extractor.compute_similarity(voiceprint, stream_emb)

            if sim > best_similarity:
                best_similarity = sim
                best_stream_idx = idx

        target_waveform = streams[best_stream_idx]
        confidence = max(0.50, min(0.99, best_similarity))

        return target_waveform, float(confidence)
