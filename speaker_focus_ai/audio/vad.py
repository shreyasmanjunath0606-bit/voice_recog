"""Voice Activity Detection (VAD) component (e.g. Silero VAD / WebRTC VAD)."""

from typing import List
import numpy as np
from speaker_focus_ai.core.types import SpeechSegment


class VoiceActivityDetector:
    """Detects speech vs. silence/noise segments in audio waveforms."""

    def __init__(self, model_name: str = "silero_vad", threshold: float = 0.5):
        self.model_name = model_name
        self.threshold = threshold
        self._model = None

    def load_model(self) -> None:
        """Load VAD model lazily."""
        pass

    def get_speech_timestamps(
        self, waveform: np.ndarray, sample_rate: int = 16000
    ) -> List[SpeechSegment]:
        """Detect speech segments in the waveform.
        
        Returns:
            List of SpeechSegment with start and end timestamps in seconds.
        """
        return []
