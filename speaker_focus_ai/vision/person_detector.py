"""Person detection and video frame extraction (Phase 1 & Phase 2).

Provides:
- Video frame decoding with millisecond-precise timestamps (Phase 1).
- YOLOv8-nano person detection accelerated on Apple Silicon MPS or CPU fallback (Phase 2).
"""

import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Generator, List, Optional, Tuple

from speaker_focus_ai.core.config import get_best_device
from speaker_focus_ai.core.types import BoundingBox


@dataclass
class VideoMetadata:
    """Metadata describing video properties."""
    fps: float
    width: int
    height: int
    total_frames: int
    duration_sec: float


def get_video_metadata(video_path: str) -> VideoMetadata:
    """Retrieve video dimensions, framerate, frame count, and duration.
    
    Args:
        video_path: Path to the video file.
        
    Returns:
        VideoMetadata object.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0.0

    cap.release()
    return VideoMetadata(
        fps=fps,
        width=width,
        height=height,
        total_frames=total_frames,
        duration_sec=duration_sec
    )


def read_video_frames(
    video_path: str,
    max_frames: Optional[int] = None
) -> Generator[Tuple[int, float, np.ndarray], None, None]:
    """Generator yielding decoded frames with accurate timestamps.
    
    Args:
        video_path: Path to the input video file.
        max_frames: Optional cap on frames to read.
        
    Yields:
        Tuple of (frame_idx, timestamp_sec, frame_bgr)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_sec = frame_idx / fps
            yield frame_idx, timestamp_sec, frame

            frame_idx += 1
            if max_frames and frame_idx >= max_frames:
                break
    finally:
        cap.release()


class PersonDetector:
    """Detects human bodies in video frames using YOLOv8-nano."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.40,
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device or get_best_device()
        self._model = None

    def load_model(self) -> None:
        """Lazily load the YOLO model onto the accelerator device."""
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.model_name)

    def detect(self, frame: np.ndarray) -> List[BoundingBox]:
        """Detect human bodies in a single frame.
        
        Args:
            frame: BGR or RGB image array of shape (H, W, 3).
            
        Returns:
            List of BoundingBox objects for detected persons.
        """
        self.load_model()

        # classes=[0] filters detection strictly to 'person' in the COCO dataset
        results = self._model(
            frame,
            classes=[0],
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False
        )

        detections: List[BoundingBox] = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                coords = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                detections.append(
                    BoundingBox(
                        x1=float(coords[0]),
                        y1=float(coords[1]),
                        x2=float(coords[2]),
                        y2=float(coords[3]),
                        confidence=conf
                    )
                )

        return detections
