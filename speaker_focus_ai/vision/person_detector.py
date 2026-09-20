"""Person detection component (e.g. YOLOv8-nano / RT-DETR / MediaPipe)."""

from typing import List
import numpy as np
from speaker_focus_ai.core.types import BoundingBox


class PersonDetector:
    """Detects human bodies in video frames."""

    def __init__(self, model_name: str = "yolov8n", device: str = "mps"):
        self.model_name = model_name
        self.device = device
        self._model = None

    def load_model(self) -> None:
        """Lazily load detection model onto specified device."""
        pass

    def detect(self, frame: np.ndarray) -> List[BoundingBox]:
        """Detect persons in a single frame.
        
        Args:
            frame: RGB/BGR image as numpy array.
            
        Returns:
            List of detected person bounding boxes.
        """
        return []
