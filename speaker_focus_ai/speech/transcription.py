"""Local speech transcription with timestamps (e.g. faster-whisper / Whisper.cpp / Whisper-timestamped)."""

from typing import List
import numpy as np
from speaker_focus_ai.core.types import TranscriptSegment


class SpeechTranscriber:
    """Transcribes audio with word/segment level timestamps."""

    def __init__(self, model_size: str = "base.en", device: str = "cpu", compute_type: str = "float32"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def load_model(self) -> None:
        """Lazily load Whisper model."""
        pass

    def transcribe(
        self, waveform: np.ndarray, sample_rate: int = 16000, speaker_id: str = "target"
    ) -> List[TranscriptSegment]:
        """Transcribe audio into timestamped segments attributed to target speaker."""
        return []
