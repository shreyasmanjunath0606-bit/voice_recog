"""Multi-person tracking component with Re-ID (e.g. ByteTrack / BoT-SORT)."""

from typing import List, Dict
import numpy as np
from speaker_focus_ai.core.types import BoundingBox, PersonTrack


class MultiPersonTracker:
    """Maintains consistent person IDs across video frames."""

    def __init__(self, tracker_type: str = "bytetrack"):
        self.tracker_type = tracker_type
        self.active_tracks: Dict[str, PersonTrack] = {}

    def update(
        self, frame_idx: int, timestamp_sec: float, detections: List[BoundingBox], frame: np.ndarray
    ) -> List[PersonTrack]:
        """Update tracker with new frame detections and associate with existing tracks."""
        return []

    def recover_lost_tracks(self) -> None:
        """Attempt track recovery for temporarily occluded or exited persons."""
        pass
