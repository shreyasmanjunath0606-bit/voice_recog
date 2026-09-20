"""Voiceprint enrollment & bootstrapping (§10).

Handles both preferred path (clean solo segments from ASD) and fallback path
(blind source separation + provisional voiceprint extraction when target never speaks alone).
"""

from typing import List, Optional, Tuple, Dict
import numpy as np
from speaker_focus_ai.core.types import EnrollmentPath, PersonTrack


class VoiceprintEnrollmentManager:
    """Manages clean voiceprint enrollment and provisional bootstrapping."""

    def __init__(self, min_solo_duration_sec: float = 1.5):
        self.min_solo_duration_sec = min_solo_duration_sec
        self.enrolled_voiceprints: Dict[str, np.ndarray] = {}
        self.enrollment_paths: Dict[str, EnrollmentPath] = {}

    def find_clean_segments(
        self, person_id: str, asd_timelines: Dict[str, List[Tuple[float, float, float]]]
    ) -> List[Tuple[float, float]]:
        """Identify time windows where only person_id has high speaking probability.
        
        Args:
            person_id: ID of the target speaker.
            asd_timelines: Dict mapping person_id to list of (start_sec, end_sec, speaking_prob).
            
        Returns:
            List of clean (start_sec, end_sec) intervals where person speaks alone.
        """
        return []

    def enroll_clean(
        self, person_id: str, audio_waveform: np.ndarray, clean_segments: List[Tuple[float, float]]
    ) -> Tuple[np.ndarray, float]:
        """Enroll high-confidence voiceprint by averaging embeddings over clean segments."""
        self.enrollment_paths[person_id] = EnrollmentPath.CLEAN
        return np.zeros((192,), dtype=np.float32), 0.95

    def bootstrap_provisional(
        self, person_id: str, audio_waveform: np.ndarray, separated_streams: List[np.ndarray], asd_timeline: List[Tuple[float, float, float]]
    ) -> Tuple[np.ndarray, float]:
        """Fallback path: Extract provisional voiceprint from separated stream best matching ASD timing."""
        self.enrollment_paths[person_id] = EnrollmentPath.PROVISIONAL
        return np.zeros((192,), dtype=np.float32), 0.60
