import unittest
from speaker_focus_ai.core.pipeline import SpeakerFocusPipeline
from speaker_focus_ai.core.types import TargetStatus, FailureReason
from speaker_focus_ai.multimodal.confidence import ConfidenceEngine

class TestSpeakerFocusPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = SpeakerFocusPipeline()
        self.confidence_engine = ConfidenceEngine()

    def test_pipeline_basic_flow(self):
        # We process a dummy video path and instruction
        result = self.pipeline.process_video("dummy.mp4", "the person in the red shirt")
        
        self.assertEqual(result.status, TargetStatus.CONFIRMED)
        self.assertEqual(result.target_person_id, "person_01")
        self.assertGreater(result.overall_confidence, 0.5)

    def test_confidence_engine_failure_states(self):
        # Test NO_SPEECH_DETECTED
        failure = self.confidence_engine.diagnose_failure(
            face_visible_duration=5.0, top_candidates_score_diff=0.5,
            total_speech_segments=0, target_asd_prob=0.8,
            separation_metric=0.9, has_enrollment_data=True
        )
        self.assertEqual(failure, FailureReason.NO_SPEECH_DETECTED)
        
        # Test AMBIGUOUS_TARGET
        failure = self.confidence_engine.diagnose_failure(
            face_visible_duration=5.0, top_candidates_score_diff=0.05,
            total_speech_segments=10, target_asd_prob=0.8,
            separation_metric=0.9, has_enrollment_data=True
        )
        self.assertEqual(failure, FailureReason.AMBIGUOUS_TARGET)

        # Test AUDIO_VIDEO_DESYNC equivalent (Target tracked but not speaking)
        failure = self.confidence_engine.diagnose_failure(
            face_visible_duration=5.0, top_candidates_score_diff=0.5,
            total_speech_segments=10, target_asd_prob=0.1,
            separation_metric=0.9, has_enrollment_data=True
        )
        self.assertEqual(failure, FailureReason.TARGET_NOT_SPEAKING)


if __name__ == '__main__':
    unittest.main()
