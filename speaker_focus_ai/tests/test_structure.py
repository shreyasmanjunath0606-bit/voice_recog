"""Unit test to verify package structure, imports, and interface definitions."""

import unittest
from speaker_focus_ai.core.types import (
    BoundingBox,
    PersonTrack,
    TargetStatus,
    EnrollmentPath,
    FailureReason,
    PipelineResult,
)
from speaker_focus_ai.core.config import PipelineConfig
from speaker_focus_ai.core.pipeline import SpeakerFocusPipeline
from speaker_focus_ai.interface.api import focus_on_person, process_video
from speaker_focus_ai.multimodal.confidence import ConfidenceEngine
from speaker_focus_ai.multimodal.target_matcher import TargetMatcher
from speaker_focus_ai.audio.enrollment import VoiceprintEnrollmentManager


class TestPackageStructure(unittest.TestCase):
    def test_imports_and_instantiation(self):
        config = PipelineConfig()
        pipeline = SpeakerFocusPipeline(config)
        self.assertIsNotNone(pipeline)

    def test_confidence_engine(self):
        engine = ConfidenceEngine()
        score = engine.calculate_confidence(
            visual_match=0.9,
            speaker_match=0.85,
            asd_speaking_prob=0.95,
            tracking_stability=0.9,
            enrollment_confidence=1.0,
            separation_quality=1.0
        )
        self.assertTrue(0.0 <= score <= 1.0)
        self.assertGreater(score, 0.8)

    def test_target_matcher_empty(self):
        matcher = TargetMatcher()
        query = matcher.parse_description("man in blue shirt")
        result = matcher.score_candidates(query, [])
        self.assertEqual(result.status, TargetStatus.NOT_FOUND)
        self.assertEqual(result.failure_reason, FailureReason.TARGET_LOST)

    def test_api_entrypoints(self):
        res = focus_on_person("person in black")
        self.assertIn(res["status"], [s.value for s in TargetStatus])


if __name__ == "__main__":
    unittest.main()
