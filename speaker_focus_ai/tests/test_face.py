"""Unit tests for Phase 3 — Face Detection & Body Association."""

import unittest
import numpy as np
import cv2

from speaker_focus_ai.vision.face_detector import FaceDetector
from speaker_focus_ai.core.types import BoundingBox, PersonTrack, FaceTrack


class TestFaceDetector(unittest.TestCase):

    def setUp(self):
        self.detector = FaceDetector(min_detection_confidence=0.5)

    def tearDown(self):
        self.detector.close()

    def test_detect_faces_on_blank_frame(self):
        """No faces should be detected on a solid color frame."""
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = self.detector.detect_faces(blank)
        self.assertIsInstance(faces, list)
        self.assertEqual(len(faces), 0)

    def test_face_body_association_valid(self):
        """A face in the upper portion of a body box should associate correctly."""
        body = BoundingBox(x1=100, y1=50, x2=300, y2=500)  # tall body
        face = BoundingBox(x1=150, y1=60, x2=250, y2=160)  # face near top

        result = FaceDetector.associate_face_with_body(face, body)
        self.assertTrue(result)

    def test_face_body_association_too_low(self):
        """A face near the bottom of a body box should NOT associate (knees aren't heads)."""
        body = BoundingBox(x1=100, y1=50, x2=300, y2=500)
        face = BoundingBox(x1=150, y1=400, x2=250, y2=480)  # near bottom

        result = FaceDetector.associate_face_with_body(face, body)
        self.assertFalse(result)

    def test_face_body_association_wrong_person(self):
        """A face far to the right should NOT associate with a person on the left."""
        body = BoundingBox(x1=50, y1=50, x2=200, y2=500)    # left side
        face = BoundingBox(x1=500, y1=60, x2=600, y2=160)   # right side

        result = FaceDetector.associate_face_with_body(face, body)
        self.assertFalse(result)

    def test_associate_faces_with_persons(self):
        """Given matching face and body positions, faces should be correctly assigned."""
        # Create two person tracks
        person1 = PersonTrack(
            person_id="person_01",
            bounding_box=BoundingBox(x1=50, y1=50, x2=200, y2=450),
            frame_idx=0, timestamp_sec=0.0, tracking_confidence=0.9
        )
        person2 = PersonTrack(
            person_id="person_02",
            bounding_box=BoundingBox(x1=400, y1=50, x2=550, y2=450),
            frame_idx=0, timestamp_sec=0.0, tracking_confidence=0.85
        )

        # Create two faces matching the person positions
        face1 = FaceTrack(
            face_id="face_001", confidence=0.95,
            bbox=BoundingBox(x1=80, y1=60, x2=170, y2=150)
        )
        face2 = FaceTrack(
            face_id="face_002", confidence=0.88,
            bbox=BoundingBox(x1=430, y1=60, x2=520, y2=150)
        )

        persons = [person1, person2]
        faces = [face1, face2]

        updated = self.detector.associate_faces_with_persons(faces, persons)

        self.assertTrue(updated[0].face_available)
        self.assertEqual(updated[0].associated_face.face_id, "face_001")

        self.assertTrue(updated[1].face_available)
        self.assertEqual(updated[1].associated_face.face_id, "face_002")

    def test_no_faces_sets_face_unavailable(self):
        """When no faces are detected, all persons should have face_available=False."""
        person = PersonTrack(
            person_id="person_01",
            bounding_box=BoundingBox(x1=50, y1=50, x2=200, y2=450),
            frame_idx=0, timestamp_sec=0.0, tracking_confidence=0.9
        )

        updated = self.detector.associate_faces_with_persons([], [person])
        self.assertFalse(updated[0].face_available)
        self.assertIsNone(updated[0].associated_face)

    def test_extract_face_crop(self):
        """Face crop extraction should return a valid numpy array."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        face = FaceTrack(
            face_id="face_001", confidence=0.9,
            bbox=BoundingBox(x1=200, y1=100, x2=300, y2=200)
        )

        crop = self.detector.extract_face_crop(frame, face, margin=0.15)
        self.assertIsNotNone(crop)
        self.assertEqual(len(crop.shape), 3)  # H, W, C
        self.assertGreater(crop.shape[0], 0)
        self.assertGreater(crop.shape[1], 0)


if __name__ == "__main__":
    unittest.main()
