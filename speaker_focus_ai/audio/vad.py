"""Voice Activity Detection (VAD) component using Silero VAD."""

from typing import List
import numpy as np
import torch
from speaker_focus_ai.core.types import SpeechSegment


class VoiceActivityDetector:
    """Detects speech vs. silence/noise segments in audio waveforms."""

    def __init__(self, model_name: str = "silero_vad", threshold: float = 0.5):
        self.model_name = model_name
        self.threshold = threshold
        self._model = None
        self._utils = None

    def load_model(self) -> None:
        """Load Silero VAD model lazily."""
        if self._model is None:
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True
            )
            self._model = model
            self._utils = utils

    def get_speech_timestamps(
        self, waveform: np.ndarray, sample_rate: int = 16000
    ) -> List[SpeechSegment]:
        """Detect speech segments in the waveform.
        
        Args:
            waveform: Float32 audio numpy array.
            sample_rate: Audio sampling rate (default 16000).
            
        Returns:
            List of SpeechSegment with start and end timestamps in seconds.
        """
        self.load_model()
        
        if len(waveform) == 0:
            return []

        # Convert numpy array to torch tensor
        audio_tensor = torch.from_numpy(waveform).float()
        
        # Get helper function from silero utils
        get_speech_timestamps_fn = self._utils[0]
        
        # Retrieve speech timestamp dicts from Silero VAD
        timestamps = get_speech_timestamps_fn(
            audio_tensor,
            self._model,
            threshold=self.threshold,
            sampling_rate=sample_rate,
            return_seconds=True
        )

        segments = [
            SpeechSegment(
                start_sec=float(ts['start']),
                end_sec=float(ts['end']),
                confidence=float(ts.get('confidence', 1.0))
            )
            for ts in timestamps
        ]

        return segments
