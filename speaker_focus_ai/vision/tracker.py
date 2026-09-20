"""Multi-person tracking with ByteTrack (Phase 2).

Maintains persistent person identities across frames, handles short-term occlusions,
and assigns normalized spatial position tags.
"""

from typing import Dict, List, Optional
import numpy as np

from speaker_focus_ai.core.config import get_best_device
from speaker_focus_ai.core.types import BoundingBox, PersonTrack


def compute_spatial_position(bbox: BoundingBox, frame_width: int) -> str:
    """Compute normalized spatial tag ('left', 'center', 'right') based on center X."""
    if frame_width <= 0:
        return "center"
    center_x = (bbox.x1 + bbox.x2) / 2.0 / frame_width
    if center_x < 0.35:
        return "left"
    elif center_x > 0.65:
        return "right"
    return "center"


class MultiPersonTracker:
    """Tracks multiple persons persistently across video frames using ByteTrack."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        tracker_config: str = "bytetrack.yaml",
        confidence_threshold: float = 0.40,
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.tracker_config = tracker_config
        self.confidence_threshold = confidence_threshold
        self.device = device or get_best_device()
        self._model = None
        self.active_tracks: Dict[str, PersonTrack] = {}

    def load_model(self) -> None:
        """Lazily load tracking model."""
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.model_name)

    def reset(self) -> None:
        """Reset internal tracker state for a new video session."""
        self.active_tracks.clear()
        if self._model is not None:
            # Reset YOLO tracker state if present
            self._model = None

    def track_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp_sec: float
    ) -> List[PersonTrack]:
        """Process a video frame and return all active person tracks with persistent IDs.
        
        Args:
            frame: Image array of shape (H, W, 3).
            frame_idx: Current frame index (0-based).
            timestamp_sec: Frame timestamp in seconds.
            
        Returns:
            List of PersonTrack objects for all currently detected and tracked persons.
        """
        self.load_model()
        frame_height, frame_width = frame.shape[:2]

        # Use ultralytics integrated ByteTrack with persistent track memory
        results = self._model.track(
            frame,
            persist=True,
            classes=[0],
            conf=self.confidence_threshold,
            tracker=self.tracker_config,
            device=self.device,
            verbose=False
        )

        current_tracks: List[PersonTrack] = []

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                coords = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                
                # Check if ByteTrack assigned a track ID
                if boxes.id is not None:
                    raw_id = int(boxes.id[i].cpu().numpy())
                    person_id = f"person_{raw_id:02d}"
                else:
                    person_id = f"person_{i+1:02d}"

                bbox = BoundingBox(
                    x1=float(coords[0]),
                    y1=float(coords[1]),
                    x2=float(coords[2]),
                    y2=float(coords[3]),
                    confidence=conf
                )

                pos = compute_spatial_position(bbox, frame_width)

                track = PersonTrack(
                    person_id=person_id,
                    bounding_box=bbox,
                    frame_idx=frame_idx,
                    timestamp_sec=timestamp_sec,
                    tracking_confidence=conf,
                    face_available=False,
                    body_visible=True,
                    position=pos
                )

                self.active_tracks[person_id] = track
                current_tracks.append(track)

        return current_tracks
