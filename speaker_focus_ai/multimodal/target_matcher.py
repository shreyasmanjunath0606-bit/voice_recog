"""Natural-language target interpretation and candidate matching (§5, §6).

Phase 5 Implementation: Parses natural descriptions into TargetQuery and calculates 
weighted candidate scores. Explicitly handles AMBIGUOUS_TARGET.
"""

from typing import List, Dict, Any
import re
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
        
        # Simple vocabulary for parsing queries
        self.colors = ["red", "blue", "green", "yellow", "black", "white", "gray", "orange", "purple", "brown"]
        self.genders = {"man": "man", "guy": "man", "boy": "man", "male": "man", 
                        "woman": "woman", "lady": "woman", "girl": "woman", "female": "woman"}
        self.positions = ["left", "right", "center", "middle"]

    def parse_description(self, instruction: str) -> TargetQuery:
        """Convert natural language query (e.g. 'man in blue shirt') to structured attributes."""
        instruction = instruction.lower().strip()
        attrs = {}
        
        # 1. Parse gender
        for word in instruction.split():
            # Remove punctuation
            word = re.sub(r'[^\w\s]', '', word)
            if word in self.genders:
                attrs["gender_hint"] = self.genders[word]
                break
                
        # 2. Parse position
        for pos in self.positions:
            if pos in instruction:
                # Map 'middle' to 'center'
                attrs["position"] = "center" if pos == "middle" else pos
                break
                
        # 3. Parse shirt and pants colors
        # A simple heuristic: color before "shirt/t-shirt/top/hoodie" is shirt color
        # color before "pants/jeans/shorts/trousers" is pants color
        
        # Find all mentioned colors
        mentioned_colors = []
        for word in instruction.split():
            word = re.sub(r'[^\w\s]', '', word)
            if word in self.colors:
                mentioned_colors.append(word)
                
        # Simple regex for shirt/pants association
        shirt_match = re.search(r'(' + '|'.join(self.colors) + r')\s+(?:shirt|t-shirt|top|hoodie|jacket|sweater)', instruction)
        pants_match = re.search(r'(' + '|'.join(self.colors) + r')\s+(?:pants|jeans|shorts|trousers)', instruction)
        
        if shirt_match:
            attrs["shirt_color"] = shirt_match.group(1)
        
        if pants_match:
            attrs["pants_color"] = pants_match.group(1)
            
        # Fallback: if only one color is mentioned and no specific garment, assume it's the shirt
        if not shirt_match and not pants_match and len(mentioned_colors) == 1:
            attrs["shirt_color"] = mentioned_colors[0]
            
        return TargetQuery(raw_text=instruction, structured_attributes=attrs)

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

        q_attrs = query.structured_attributes
        
        scored: List[CandidateScore] = []
        for cand in candidates:
            c_attrs = cand.clothing_attributes
            
            # Feature matching (1.0 for match, 0.0 for mismatch/unknown)
            # 1. Shirt Color
            score_shirt_color = 0.0
            if "shirt_color" in q_attrs and c_attrs.shirt_color:
                if q_attrs["shirt_color"] == c_attrs.shirt_color:
                    score_shirt_color = 1.0
                    
            # 2. Pants Color
            score_pants_color = 0.0
            if "pants_color" in q_attrs and c_attrs.pants_color:
                if q_attrs["pants_color"] == c_attrs.pants_color:
                    score_pants_color = 1.0
                    
            # 3. Position
            score_position = 0.0
            if "position" in q_attrs and cand.position:
                if q_attrs["position"] == cand.position:
                    score_position = 1.0
                    
            # 4. Gender (Not strictly extracted by vision yet, but we put placeholder matching)
            score_gender = 0.0
            if "gender_hint" in q_attrs and c_attrs.gender_hint:
                if q_attrs["gender_hint"] == c_attrs.gender_hint:
                    score_gender = 1.0
                    
            # 5. Face Available (We prefer candidates that have faces since we need ASD)
            score_face = 1.0 if cand.face_available else 0.0
            
            # 6. Shirt Type (Optional/Future ML feature)
            score_shirt_type = 0.0
            
            # Combine into Visual Match Score (which is sum of matched components divided by requested components)
            # We use the config weights to compute the overall visual match score.
            # But the spec says: "Calculate weighted candidate score: Shirt color (35%), ... Face available (15%)"
            
            overall = (
                score_shirt_color * self.weights.shirt_color
                + score_shirt_type * self.weights.shirt_type
                + score_pants_color * self.weights.pants_color
                + score_position * self.weights.position
                + score_gender * self.weights.gender_hint
                + score_face * self.weights.face_info
            )
            
            # Boost score slightly by tracking confidence so stable tracks win ties
            overall += (cand.tracking_confidence * 0.01)
            
            scored.append(
                CandidateScore(
                    person_id=cand.person_id,
                    visual_match_score=overall, # In this simplified model, visual match is the overall
                    tracking_score=cand.tracking_confidence,
                    face_match_score=score_face,
                    position_score=score_position,
                    overall_target_score=overall
                )
            )

        scored.sort(key=lambda x: x.overall_target_score, reverse=True)

        # Check ambiguity rule: diff between top 2 <= ambiguity_threshold
        if len(scored) > 1:
            diff = scored[0].overall_target_score - scored[1].overall_target_score
            if diff <= self.ambiguity_threshold:
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
