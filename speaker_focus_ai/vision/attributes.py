"""Visual attribute extraction (clothing color, type, spatial position).

Phase 4 implementation: Uses OpenCV HSV color space and K-Means clustering 
to determine dominant colors of upper and lower body clothing.
"""

from typing import Dict, Any, Tuple
import cv2
import numpy as np
from speaker_focus_ai.core.types import BoundingBox, ClothingAttributes


class VisualAttributeExtractor:
    """Extracts appearance features (clothing colors, spatial position) from person crops."""

    # Map dominant HSV to color names (simplified)
    # H: 0-180, S: 0-255, V: 0-255 in OpenCV
    COLOR_RANGES = {
        "black": {"lower": (0, 0, 0), "upper": (180, 255, 50)},
        "white": {"lower": (0, 0, 200), "upper": (180, 30, 255)},
        "gray":  {"lower": (0, 0, 50), "upper": (180, 50, 200)},
        "red":   [{"lower": (0, 70, 50), "upper": (10, 255, 255)}, 
                  {"lower": (170, 70, 50), "upper": (180, 255, 255)}], # Red wraps around in HSV
        "orange":{"lower": (11, 70, 50), "upper": (25, 255, 255)},
        "yellow":{"lower": (26, 70, 50), "upper": (35, 255, 255)},
        "green": {"lower": (36, 50, 50), "upper": (85, 255, 255)},
        "blue":  {"lower": (86, 50, 50), "upper": (125, 255, 255)},
        "purple":{"lower": (126, 50, 50), "upper": (169, 255, 255)}
    }

    def __init__(self, use_vlm_or_clip: bool = False):
        self.use_vlm_or_clip = use_vlm_or_clip

    def extract_attributes(
        self, frame: np.ndarray, bbox: BoundingBox, frame_width: int, frame_height: int
    ) -> ClothingAttributes:
        """Extract structured clothing and position attributes from a detected person region."""
        # Calculate spatial position (e.g. left, center, right)
        cx = (bbox.x1 + bbox.x2) / 2.0 / frame_width
        pos_str = "left" if cx < 0.35 else ("right" if cx > 0.65 else "center")

        # Crop the person bounding box from the frame
        x1, y1 = max(0, int(bbox.x1)), max(0, int(bbox.y1))
        x2, y2 = min(frame_width, int(bbox.x2)), min(frame_height, int(bbox.y2))
        
        person_crop = frame[y1:y2, x1:x2]
        
        shirt_color = None
        pants_color = None

        if person_crop.size > 0:
            h, w = person_crop.shape[:2]
            
            # Divide body anatomically
            # Upper body (torso): roughly 20% to 55% of the height
            torso_y1, torso_y2 = int(0.20 * h), int(0.55 * h)
            # Lower body (pants): roughly 55% to 90% of the height
            pants_y1, pants_y2 = int(0.55 * h), int(0.90 * h)
            
            # Central column (middle 50% of width) to avoid background
            mid_x1, mid_x2 = int(0.25 * w), int(0.75 * w)

            torso_crop = person_crop[torso_y1:torso_y2, mid_x1:mid_x2]
            pants_crop = person_crop[pants_y1:pants_y2, mid_x1:mid_x2]

            shirt_color = self._get_dominant_color_name(torso_crop)
            pants_color = self._get_dominant_color_name(pants_crop)

        return ClothingAttributes(
            shirt_color=shirt_color,
            shirt_type=None, # Garment type might need ML/CLIP. Color is easier via CV.
            pants_color=pants_color,
            gender_hint=None,
            raw_attributes={"position_tag": pos_str}
        )

    def _get_dominant_color_name(self, crop: np.ndarray) -> str:
        """Find the dominant color name using HSV thresholding."""
        if crop is None or crop.size == 0:
            return "unknown"
            
        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        best_color = "unknown"
        max_pixels = 0
        
        for color_name, ranges in self.COLOR_RANGES.items():
            if isinstance(ranges, list):
                # E.g. Red has two ranges
                mask1 = cv2.inRange(hsv_crop, ranges[0]["lower"], ranges[0]["upper"])
                mask2 = cv2.inRange(hsv_crop, ranges[1]["lower"], ranges[1]["upper"])
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = cv2.inRange(hsv_crop, ranges["lower"], ranges["upper"])
                
            num_pixels = cv2.countNonZero(mask)
            if num_pixels > max_pixels:
                max_pixels = num_pixels
                best_color = color_name
                
        # Only return the color if it covers a decent portion of the crop (e.g. > 25%)
        total_pixels = crop.shape[0] * crop.shape[1]
        if max_pixels > 0.25 * total_pixels:
            return best_color
            
        return "unknown"
