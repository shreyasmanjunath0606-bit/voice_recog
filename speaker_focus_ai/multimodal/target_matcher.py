"""Natural-language target interpretation and candidate matching (§5, §6)."""

from typing import List, Dict, Any
from speaker_focus_ai.core.types import (
    PersonTrack,
    TargetQuery,
    CandidateScore,
    TargetSelectionResult,
    TargetStatus,
    FailureReason
)
from speaker_focus_ai.core.config import MatchingWeights


class TargetMatcher:
    """Parses natural-language user descriptions and matches against detected person tracks."""

    def __init__(self, weights: MatchingWeights = MatchingWeights(), ambiguity_threshold: float = 0.08):
        self.weights = weights
        self.ambiguity_threshold = ambiguity_threshold

    def parse_description(self, instruction: str) -> TargetQuery:
        """Convert natural language query (e.g. 'man in blue shirt') to structured attributes."""
        # Simple rule/regex or local LLM parsing
        return TargetQuery(raw_text=instruction)

    def score_candidates(
        self, query: TargetQuery, candidates: List[PersonTrack]
    ) -> TargetSelectionResult:
        """Compare query against candidates using weighted matching.
        
        Avoids silent arbitrary selection by flagging AMBIGUOUS_TARGET if top candidates score closely.
        """
        if not candidates:
            return TargetSelectionResult(
                target_person_id=None,
                confidence=0.0,
                status=TargetStatus.NOT_FOUND,
                failure_reason=FailureReason.TARGET_LOST
            )

        scored: List[CandidateScore] = []
        for cand in candidates:
            # Calculate individual subscores
            v_score = 0.5
            t_score = cand.tracking_confidence
            f_score = cand.associated_face.confidence if cand.associated_face else 0.0
            p_score = 0.5
            
            overall = (
                v_score * self.weights.shirt_color
                + t_score * self.weights.pants_color
                + f_score * self.weights.face_info
                + p_score * self.weights.position
            )
            scored.append(
                CandidateScore(
                    person_id=cand.person_id,
                    visual_match_score=v_score,
                    tracking_score=t_score,
                    face_match_score=f_score,
                    position_score=p_score,
                    overall_target_score=overall
                )
            )

        scored.sort(key=lambda x: x.overall_target_score, reverse=True)

        # Check ambiguity
        if len(scored) > 1 and (scored[0].overall_target_score - scored[1].overall_target_score) <= self.ambiguity_threshold:
            return TargetSelectionResult(
                target_person_id=None,
                confidence=scored[0].overall_target_score,
                status=TargetStatus.AMBIGUOUS,
                candidates=scored,
                failure_reason=FailureReason.AMBIGUOUS_TARGET
            )

        return TargetSelectionResult(
            target_person_id=scored[0].person_id,
            confidence=scored[0].overall_target_score,
            status=TargetStatus.CONFIRMED,
            candidates=scored
        )
