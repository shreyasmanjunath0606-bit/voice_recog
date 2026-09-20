"""Unit tests for Phase 1 (Video Reading) and Phase 2 (Person Detection & Tracking)."""

import os
import tempfile
import unittest
import cv2
import numpy as np

from speaker_focus_ai.vision.person_detector import (
    VideoMetadata,
    get_video_metadata,
    read_video_frames,
    PersonDetector,
)
from speaker_focus_ai.vision.tracker import MultiPersonTracker, compute_spatial_position
from speaker_focus_ai.core.types import BoundingBox


class TestVisionPipeline(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory and synthetic MP4 video
        self.temp_dir = tempfile.TemporaryDirectory()
        self.video_path = os.path.join(self.temp_dir.name, "test_synthetic.mp4")
        self._generate_synthetic_video(self.video_path, fps=30.0, num_frames=30)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _generate_synthetic_video(self, path: str, fps: float = 30.0, num_frames: int = 30):
        """Generate a short video with moving shapes to test video decoding."""
        width, height = 320, 240
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(path, fourcc, fps, (width, height))
        for i in range(num_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            # Draw moving rectangle
            x = int(50 + (i * 5) % (width - 100))
            cv2.rectangle(frame, (x, 50), (x + 40, 150), (0, 255, 0), -1)
            out.write(frame)
        out.release()

    def test_video_metadata(self):
        meta = get_video_metadata(self.video_path)
        self.assertIsInstance(meta, VideoMetadata)
        self.assertAlmostEqual(meta.fps, 30.0, delta=1.0)
        self.assertEqual(meta.total_frames, 30)
        self.assertAlmostEqual(meta.duration_sec, 1.0, delta=0.1)

    def test_read_video_frames(self):
        frames = list(read_video_frames(self.video_path))
        self.assertEqual(len(frames), 30)
        
        frame_idx, timestamp_sec, img = frames[15]
        self.assertEqual(frame_idx, 15)
        self.assertAlmostEqual(timestamp_sec, 0.5, delta=0.05)
        self.assertEqual(img.shape, (240, 320, 3))

    def test_spatial_position_calculation(self):
        frame_width = 1000
        left_box = BoundingBox(x1=50, y1=100, x2=200, y2=400)
        center_box = BoundingBox(x1=400, y1=100, x2=600, y2=400)
        right_box = BoundingBox(x1=800, y1=100, x2=950, y2=400)

        self.assertEqual(compute_spatial_position(left_box, frame_width), "left")
        self.assertEqual(compute_spatial_position(center_box, frame_width), "center")
        self.assertEqual(compute_spatial_position(right_box, frame_width), "right")

    def test_person_detector_inference(self):
        detector = PersonDetector()
        dummy_frame = np.zeros((320, 320, 3), dtype=np.uint8)
        # Should run inference cleanly without errors
        detections = detector.detect(dummy_frame)
        self.assertIsInstance(detections, list)

    def test_tracker_inference_and_reset(self):
        tracker = MultiPersonTracker()
        dummy_frame = np.zeros((320, 320, 3), dtype=np.uint8)
        tracks = tracker.track_frame(dummy_frame, frame_idx=0, timestamp_sec=0.0)
        self.assertIsInstance(tracks, list)
        tracker.reset()
        self.assertEqual(len(tracker.active_tracks), 0)


if __name__ == "__main__":
    unittest.main()
