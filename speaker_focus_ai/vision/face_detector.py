"""Face detection and face-to-body association (Phase 3).

Uses OpenCV's built-in YuNet DNN face detector (fast, reliable, no GPU issues).
Associates each detected face with the closest anatomically valid PersonTrack.
"""

import os
from typing import List, Optional
import cv2
import numpy as np

from speaker_focus_ai.core.types import FaceTrack, BoundingBox, PersonTrack

# Default YuNet ONNX model path relative to this package
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models", "face_detection_yunet.onnx"
)


class FaceDetector:
    """Detects faces using OpenCV YuNet and associates them with tracked person bodies."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        min_detection_confidence: float = 0.5,
        input_size: tuple = (640, 480)
    ):
        self.model_path = model_path or _DEFAULT_MODEL_PATH
        self.min_detection_confidence = min_detection_confidence
        self.input_size = input_size
        self._detector = None
        self._face_counter = 0

    def load_model(self) -> None:
        """Lazily initialize OpenCV YuNet face detector."""
        if self._detector is not None:
            return

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Face detection model not found at: {self.model_path}\n"
                "Download it with:\n"
                "  curl -L 'https://github.com/opencv/opencv_zoo/raw/main/"
                "models/face_detection_yunet/face_detection_yunet_2023mar.onnx'"
                " -o speaker_focus_ai/models/face_detection_yunet.onnx"
            )

        self._detector = cv2.FaceDetectorYN.create(
            self.model_path,
            "",
            self.input_size,
            self.min_detection_confidence,
            0.3,   # NMS threshold
            5000   # top_k
        )

    def detect_faces(self, frame: np.ndarray) -> List[FaceTrack]:
        """Detect all faces in a single BGR frame.

        Args:
            frame: BGR image array of shape (H, W, 3).

        Returns:
            List of FaceTrack objects with bounding boxes and confidence scores.
        """
        self.load_model()
        h, w = frame.shape[:2]

        # Update input size if frame dimensions differ from initial config
        self._detector.setInputSize((w, h))

        _, raw_detections = self._detector.detect(frame)

        faces: List[FaceTrack] = []
        if raw_detections is not None:
            for det in raw_detections:
                # YuNet output per detection: [x, y, w, h, ..., score]
                x1 = max(0.0, float(det[0]))
                y1 = max(0.0, float(det[1]))
                x2 = min(float(w), float(det[0] + det[2]))
                y2 = min(float(h), float(det[1] + det[3]))
                conf = float(det[-1])

                self._face_counter += 1
                face_id = f"face_{self._face_counter:03d}"

                faces.append(
                    FaceTrack(
                        face_id=face_id,
                        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf),
                        confidence=conf,
                        embedding=None
                    )
                )

        return faces

    def extract_face_crop(
        self, frame: np.ndarray, face: FaceTrack, margin: float = 0.15
    ) -> Optional[np.ndarray]:
        """Extract a padded face crop from the frame for downstream ASD models.

        Args:
            frame: BGR image array.
            face: FaceTrack with bounding box.
            margin: Fractional padding around the face (e.g. 0.15 = 15% extra).

        Returns:
            Cropped face image or None if the crop is invalid.
        """
        h, w = frame.shape[:2]
        bw = face.bbox.x2 - face.bbox.x1
        bh = face.bbox.y2 - face.bbox.y1

        x1 = max(0, int(face.bbox.x1 - bw * margin))
        y1 = max(0, int(face.bbox.y1 - bh * margin))
        x2 = min(w, int(face.bbox.x2 + bw * margin))
        y2 = min(h, int(face.bbox.y2 + bh * margin))

        if x2 <= x1 or y2 <= y1:
            return None

        return frame[y1:y2, x1:x2].copy()

    @staticmethod
    def associate_face_with_body(
        face_bbox: BoundingBox, person_bbox: BoundingBox, tolerance: float = 0.15
    ) -> bool:
        """Check if a face bounding box is anatomically within a person's body bounding box.

        The face should be:
        - Horizontally centered within the body box (with tolerance for head tilt).
        - Vertically within the upper 45% of the body box.

        Args:
            face_bbox: The detected face bounding box.
            person_bbox: The tracked person body bounding box.
            tolerance: Fractional horizontal slack (0.15 = 15% of body width).

        Returns:
            True if the face anatomically belongs to this person.
        """
        body_w = person_bbox.x2 - person_bbox.x1
        body_h = person_bbox.y2 - person_bbox.y1
        slack = body_w * tolerance

        # Face center must be horizontally within the body box (with slack)
        face_cx = (face_bbox.x1 + face_bbox.x2) / 2.0
        if face_cx < (person_bbox.x1 - slack) or face_cx > (person_bbox.x2 + slack):
            return False

        # Face center must be in the upper 45% of the body
        upper_limit = person_bbox.y1 + body_h * 0.45
        face_cy = (face_bbox.y1 + face_bbox.y2) / 2.0
        if face_cy > upper_limit:
            return False

        return True

    def associate_faces_with_persons(
        self, faces: List[FaceTrack], persons: List[PersonTrack]
    ) -> List[PersonTrack]:
        """Match each detected face with the best-fitting person track.

        Each face is assigned to at most one person, and each person gets
        at most one face (the highest confidence one if multiple match).

        Args:
            faces: Detected faces in this frame.
            persons: Tracked person bodies in this frame.

        Returns:
            Updated list of PersonTrack objects with face associations filled in.
        """
        # Reset face associations for this frame
        for person in persons:
            person.face_available = False
            person.associated_face = None

        if not faces or not persons:
            return persons

        # For each person, find the best matching face
        assigned_faces = set()
        for person in persons:
            best_face: Optional[FaceTrack] = None
            best_conf = -1.0

            for face in faces:
                if face.face_id in assigned_faces:
                    continue

                if self.associate_face_with_body(face.bbox, person.bounding_box):
                    if face.confidence > best_conf:
                        best_face = face
                        best_conf = face.confidence

            if best_face is not None:
                person.face_available = True
                person.associated_face = best_face
                assigned_faces.add(best_face.face_id)

        return persons

    def close(self) -> None:
        """Release detector resources."""
        self._detector = None
