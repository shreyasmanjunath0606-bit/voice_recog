"""Unit tests for Phase 4 — Visual Attribute Extraction."""

import unittest
import numpy as np
import cv2

from speaker_focus_ai.vision.attributes import VisualAttributeExtractor
from speaker_focus_ai.core.types import BoundingBox, ClothingAttributes


class TestVisualAttributeExtractor(unittest.TestCase):

    def setUp(self):
        self.extractor = VisualAttributeExtractor(use_vlm_or_clip=False)
        self.frame_w, self.frame_h = 640, 480
        
        # Create a synthetic frame (gray background)
        self.frame = np.full((self.frame_h, self.frame_w, 3), 128, dtype=np.uint8)

    def test_extract_spatial_position(self):
        """Test if spatial positions (left, right, center) are extracted correctly."""
        
        # Left person
        bbox_left = BoundingBox(x1=10, y1=100, x2=110, y2=400)
        attr_left = self.extractor.extract_attributes(self.frame, bbox_left, self.frame_w, self.frame_h)
        self.assertEqual(attr_left.raw_attributes.get("position_tag"), "left")
        
        # Center person
        bbox_center = BoundingBox(x1=270, y1=100, x2=370, y2=400)
        attr_center = self.extractor.extract_attributes(self.frame, bbox_center, self.frame_w, self.frame_h)
        self.assertEqual(attr_center.raw_attributes.get("position_tag"), "center")
        
        # Right person
        bbox_right = BoundingBox(x1=500, y1=100, x2=600, y2=400)
        attr_right = self.extractor.extract_attributes(self.frame, bbox_right, self.frame_w, self.frame_h)
        self.assertEqual(attr_right.raw_attributes.get("position_tag"), "right")

    def test_extract_clothing_colors(self):
        """Test extraction of shirt and pants colors on a synthetic person."""
        
        # BBox for a person in the center
        bbox = BoundingBox(x1=270, y1=100, x2=370, y2=400)
        
        # Person height is 300 (400-100). Torso is 20-55% -> 60 to 165 px from top (160 to 265 in frame)
        # Pants is 55-90% -> 165 to 270 px from top (265 to 370 in frame)
        # Width is 100 (370-270). Center 50% is 25-75 -> 25 to 75 px from left (295 to 345 in frame)
        
        # Paint the torso red (BGR)
        cv2.rectangle(self.frame, (270, 160), (370, 265), (0, 0, 255), -1) 
        
        # Paint the pants blue (BGR)
        cv2.rectangle(self.frame, (270, 265), (370, 370), (255, 0, 0), -1)

        attr = self.extractor.extract_attributes(self.frame, bbox, self.frame_w, self.frame_h)
        
        self.assertEqual(attr.shirt_color, "red")
        self.assertEqual(attr.pants_color, "blue")
        
    def test_unknown_color(self):
        """Test that unknown/ambiguous colors return 'unknown'."""
        
        bbox = BoundingBox(x1=270, y1=100, x2=370, y2=400)
        
        # Paint torso with thin bands of colors so no single color exceeds 25%
        crop = self.frame[100:400, 270:370]
        colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 0, 255)]
        
        for i in range(0, 300, 20):
            color = colors[(i // 20) % len(colors)]
            crop[i:i+20, :] = color

        attr = self.extractor.extract_attributes(self.frame, bbox, self.frame_w, self.frame_h)
        self.assertEqual(attr.shirt_color, "unknown")
        self.assertEqual(attr.pants_color, "unknown")


    def test_invalid_bbox(self):
        """Test behavior when the bounding box is invalid/out of bounds."""
        # Box completely outside
        bbox = BoundingBox(x1=700, y1=700, x2=800, y2=800)
        attr = self.extractor.extract_attributes(self.frame, bbox, self.frame_w, self.frame_h)
        self.assertEqual(attr.shirt_color, None)
        self.assertEqual(attr.pants_color, None)

if __name__ == "__main__":
    unittest.main()
