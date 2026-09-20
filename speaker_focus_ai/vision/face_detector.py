"""Face detection and embedding extraction component (e.g. MediaPipe Face / RetinaFace / InsightFace)."""

from typing import List, Optional
import numpy as np
from speaker_focus_ai.core.types import FaceTrack, BoundingBox


class FaceDetector:
    """Detects faces and computes facial embeddings."""

    def __init__(self, model_name: str = "mediapipe", device: str = "mps"):
        self.model_name = model_name
        self.device = device
        self._model = None

    def load_model(self) -> None:
        """Lazily load face detection and recognition models."""
        pass

    def detect_faces(self, frame: np.ndarray) -> List[FaceTrack]:
        """Detect faces and facial landmarks in a frame."""
        return []

    def associate_face_with_body(
        self, face_bbox: BoundingBox, person_bbox: BoundingBox
    ) -> bool:
        """Check if face bounding box is anatomically within or associated with person bounding box."""
        return (
            face_bbox.x1 >= person_bbox.x1
            and face_bbox.x2 <= person_bbox.x2
            and face_bbox.y1 >= person_bbox.y1
            and face_bbox.y2 <= person_bbox.y1 + 0.5 * (person_bbox.y2 - person_bbox.y1)
        )
