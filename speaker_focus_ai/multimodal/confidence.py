"""Confidence scoring engine combining measurable signals (§22).

Calculates transparent, non-fabricated confidence from:
- visual_match
- face_match
- speaker_match
- asd_speaking_probability
- tracking_stability
- separation_quality
- enrollment_confidence (clean vs provisional)
"""

from typing import Dict, Any
from speaker_focus_ai.core.config import ConfidenceWeights


class ConfidenceEngine:
    """Calculates overall target speaker confidence from multiple pipeline signals."""

    def __init__(self, weights: ConfidenceWeights = ConfidenceWeights()):
        self.weights = weights

    def calculate_confidence(
        self,
        visual_match: float,
        speaker_match: float,
        asd_speaking_prob: float,
        tracking_stability: float,
        enrollment_confidence: float,
        separation_quality: float = 1.0,
    ) -> float:
        """Combine signals into a final confidence score.
        
        Note: V1 uses weighted combination with separation penalty.
        Documented in ARCHITECTURE.md as a known v1 simplification.
        """
        raw_score = (
            visual_match * self.weights.visual_match
            + speaker_match * self.weights.speaker_match
            + asd_speaking_prob * self.weights.asd_speaking_prob
            + tracking_stability * self.weights.tracking_stability
            + enrollment_confidence * self.weights.enrollment_confidence
        )
        
        # Scale by separation quality
        final_score = raw_score * min(1.0, max(0.2, separation_quality))
        return float(min(1.0, max(0.0, final_score)))

    def diagnose_failure(
        self,
        face_visible_duration: float,
        top_candidates_score_diff: float,
        total_speech_segments: int,
        target_asd_prob: float,
        separation_metric: float,
        has_enrollment_data: bool
    ):
        """Diagnose pipeline failures based on the Failure State Handling Contract (§23)."""
        from speaker_focus_ai.core.types import FailureReason
        
        if not has_enrollment_data:
            return FailureReason.INSUFFICIENT_ENROLLMENT_DATA
            
        if total_speech_segments == 0:
            return FailureReason.NO_SPEECH_DETECTED
            
        if top_candidates_score_diff <= 0.08:
            return FailureReason.AMBIGUOUS_TARGET
            
        if face_visible_duration < 3.0: # Face not visible for > 3 seconds logic (simplified)
            return FailureReason.FACE_NOT_VISIBLE
            
        if target_asd_prob < 0.15:
            return FailureReason.TARGET_NOT_SPEAKING
            
        if separation_metric < 0.35:
            return FailureReason.LOW_SEPARATION_CONFIDENCE
            
        return None
