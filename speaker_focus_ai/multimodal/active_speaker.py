"""Audio-Visual Active Speaker Detection (ASD) (§8).

End-to-end model (e.g. TalkNet-ASD / Light-ASD) correlating face crop sequence
with audio waveform to produce per-identity speaking probability over time.
"""

from typing import List, Dict, Tuple
import numpy as np


class ActiveSpeakerDetector:
    """Detects speaking status by jointly modeling lip/mouth motion and audio spectrogram."""

    def __init__(self, model_name: str = "talknet_asd", device: str = "mps"):
        self.model_name = model_name
        self.device = device
        self._model = None

    def load_model(self) -> None:
        """Lazily load ASD audio-visual model."""
        pass

    def compute_speaking_probability(
        self,
        face_crops: List[np.ndarray],
        audio_window: np.ndarray,
        fps: float = 25.0,
        sample_rate: int = 16000
    ) -> float:
        """Compute speaking probability for a given face crop sequence and audio window.
        
        Returns:
            Speaking probability float in [0.0, 1.0].
        """
        return 0.0

    def generate_speaking_timeline(
        self,
        person_tracks: Dict[str, List[Tuple[float, np.ndarray]]],
        audio_waveform: np.ndarray,
        sample_rate: int = 16000
    ) -> Dict[str, List[Tuple[float, float, float]]]:
        """Generate speaking probability timeline for each tracked identity over the video duration.
        
        Returns:
            Dict mapping person_id to list of (start_sec, end_sec, speaking_probability).
        """
        return {}
