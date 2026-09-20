"""Debug overlay visualizer (§15).

Renders bounding boxes, face boxes, speaking status, target indicator,
and telemetry overlay onto video frames.
"""

from typing import List, Optional
import numpy as np
from speaker_focus_ai.core.types import PersonTrack


class DebugVisualizer:
    """Draws diagnostic overlays on video frames for debugging."""

    def __init__(self, output_video_path: Optional[str] = None):
        self.output_video_path = output_video_path

    def draw_frame_overlay(
        self,
        frame: np.ndarray,
        tracks: List[PersonTrack],
        target_person_id: Optional[str],
        telemetry: dict
    ) -> np.ndarray:
        """Annotate frame with tracking boxes, speaking probability, target marker, and HUD."""
        # Returns annotated frame copy
        return frame.copy()
