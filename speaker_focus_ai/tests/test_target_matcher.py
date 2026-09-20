"""Unit tests for Phase 5 — Target Matcher."""

import unittest
from speaker_focus_ai.multimodal.target_matcher import TargetMatcher
from speaker_focus_ai.core.types import (
    PersonTrack, 
    BoundingBox, 
    ClothingAttributes, 
    TargetStatus,
    FailureReason
)
from speaker_focus_ai.core.config import MatchingWeights


class TestTargetMatcher(unittest.TestCase):

    def setUp(self):
        self.matcher = TargetMatcher()

    def test_parse_description(self):
        """Test natural language parsing for colors, positions, gender."""
        
        # Test 1: Simple color
        q = self.matcher.parse_description("the man in the blue shirt")
        self.assertEqual(q.structured_attributes.get("shirt_color"), "blue")
        self.assertEqual(q.structured_attributes.get("gender_hint"), "man")
        
        # Test 2: Position and fallback color
        q2 = self.matcher.parse_description("person on the left in red")
        self.assertEqual(q2.structured_attributes.get("shirt_color"), "red")
        self.assertEqual(q2.structured_attributes.get("position"), "left")
        
        # Test 3: Shirt and pants explicit
        q3 = self.matcher.parse_description("guy wearing a green hoodie and black jeans")
        self.assertEqual(q3.structured_attributes.get("shirt_color"), "green")
        self.assertEqual(q3.structured_attributes.get("pants_color"), "black")
        self.assertEqual(q3.structured_attributes.get("gender_hint"), "man")

    def test_score_candidates_clear_winner(self):
        """Test matching when there is a clearly dominant candidate."""
        query = self.matcher.parse_description("the guy in the blue shirt on the left")
        
        # Candidate 1: matches perfectly
        cand1 = PersonTrack(
            person_id="person_01",
            bounding_box=BoundingBox(0,0,10,10),
            frame_idx=0, timestamp_sec=0.0, tracking_confidence=0.9,
            position="left",
            face_available=True,
            clothing_attributes=ClothingAttributes(shirt_color="blue", gender_hint="man")
        )
        
        # Candidate 2: wrong color, wrong position
        cand2 = PersonTrack(
            person_id="person_02",
            bounding_box=BoundingBox(0,0,10,10),
            frame_idx=0, timestamp_sec=0.0, tracking_confidence=0.9,
            position="right",
            face_available=True,
            clothing_attributes=ClothingAttributes(shirt_color="red", gender_hint="man")
        )
        
        result = self.matcher.score_candidates(query, [cand1, cand2])
        self.assertEqual(result.status, TargetStatus.CONFIRMED)
        self.assertEqual(result.target_person_id, "person_01")

    def test_score_candidates_ambiguous(self):
        """Test ambiguity rule (§6) when top 2 candidates are within threshold."""
        # Query: red shirt
        query = self.matcher.parse_description("person in red shirt")
        
        # Both candidates have a red shirt!
        cand1 = PersonTrack(
            person_id="person_01",
            bounding_box=BoundingBox(0,0,10,10),
            frame_idx=0, timestamp_sec=0.0, tracking_confidence=0.9,
            face_available=True,
            clothing_attributes=ClothingAttributes(shirt_color="red")
        )
        
        cand2 = PersonTrack(
            person_id="person_02",
            bounding_box=BoundingBox(0,0,10,10),
            frame_idx=0, timestamp_sec=0.0, tracking_confidence=0.88, # slightly lower tracking conf
            face_available=True,
            clothing_attributes=ClothingAttributes(shirt_color="red")
        )
        
        result = self.matcher.score_candidates(query, [cand1, cand2])
        
        # They should have the same base score, diff is 0.9*0.01 - 0.88*0.01 = 0.0002 <= 0.08
        self.assertEqual(result.status, TargetStatus.AMBIGUOUS)
        self.assertIsNone(result.target_person_id)
        self.assertEqual(result.failure_reason, FailureReason.AMBIGUOUS_TARGET)

if __name__ == "__main__":
    unittest.main()
