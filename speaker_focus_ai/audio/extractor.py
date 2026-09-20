"""Audio extraction from video files using librosa and soundfile."""

import os
from typing import Tuple
import numpy as np
import soundfile as sf
import librosa


class AudioExtractor:
    """Extracts high-fidelity audio tracks from video containers."""

    def __init__(self, target_sample_rate: int = 16000):
        self.target_sample_rate = target_sample_rate

    def extract_waveform(self, video_path: str) -> Tuple[np.ndarray, int]:
        """Extract audio waveform as a normalized float32 mono array at target sample rate.
        
        Args:
            video_path: Absolute path to video/audio file.
            
        Returns:
            Tuple of (waveform_numpy_array, sample_rate).
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        try:
            # Load audio using librosa (handles resampling and mono conversion)
            waveform, sr = librosa.load(
                video_path,
                sr=self.target_sample_rate,
                mono=True,
                dtype=np.float32
            )
        except Exception as e:
            # Fallback to soundfile if librosa fails on raw audio files
            try:
                waveform, sr = sf.read(video_path, dtype='float32')
                if waveform.ndim > 1:
                    waveform = np.mean(waveform, axis=1)
                if sr != self.target_sample_rate:
                    waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.target_sample_rate)
                    sr = self.target_sample_rate
            except Exception as sf_err:
                raise RuntimeError(f"Failed to extract audio from {video_path}: {e} / {sf_err}")

        # Ensure float32 [-1.0, 1.0] range
        max_val = np.max(np.abs(waveform))
        if max_val > 1.0:
            waveform = waveform / max_val

        return waveform.astype(np.float32), self.target_sample_rate

    def save_wav(self, waveform: np.ndarray, sample_rate: int, output_path: str) -> str:
        """Save waveform array as a 16-bit PCM WAV file.
        
        Args:
            waveform: Float32 audio numpy array.
            sample_rate: Target sampling rate (e.g. 16000).
            output_path: Destination path.
            
        Returns:
            Absolute path to saved WAV file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        sf.write(output_path, waveform, sample_rate, subtype='PCM_16')
        return os.path.abspath(output_path)
