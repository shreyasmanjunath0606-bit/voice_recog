"""Visual attribute extraction (clothing color, type, spatial position)."""

from typing import Dict, Any
import numpy as np
from speaker_focus_ai.core.types import BoundingBox, ClothingAttributes


class VisualAttributeExtractor:
    """Extracts appearance features (clothing colors, garment types, spatial position) from person crops."""

    def __init__(self, use_vlm_or_clip: bool = True):
        self.use_vlm_or_clip = use_vlm_or_clip

    def extract_attributes(
        self, frame: np.ndarray, bbox: BoundingBox, frame_width: int, frame_height: int
    ) -> ClothingAttributes:
        """Extract structured clothing and position attributes from a detected person region."""
        # Calculate spatial position (e.g. left, center, right, closest to camera)
        cx = (bbox.x1 + bbox.x2) / 2.0 / frame_width
        pos_str = "left" if cx < 0.35 else ("right" if cx > 0.65 else "center")
        
        return ClothingAttributes(
            shirt_color=None,
            shirt_type=None,
            pants_color=None,
            gender_hint=None,
            raw_attributes={"position_tag": pos_str}
        )
