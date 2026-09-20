"""Core data types and schemas for speaker_focus_ai."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class TargetStatus(str, Enum):
    CONFIRMED = "confirmed"
    AMBIGUOUS = "ambiguous"
    NOT_FOUND = "not_found"


class EnrollmentPath(str, Enum):
    CLEAN = "clean"
    PROVISIONAL = "provisional"
    FAILED = "failed"


class FailureReason(str, Enum):
    FACE_NOT_VISIBLE = "FACE_NOT_VISIBLE"
    AMBIGUOUS_TARGET = "AMBIGUOUS_TARGET"
    NO_SPEECH_DETECTED = "NO_SPEECH_DETECTED"
    TARGET_NOT_SPEAKING = "TARGET_NOT_SPEAKING"
    LOW_SEPARATION_CONFIDENCE = "LOW_SEPARATION_CONFIDENCE"
    LOW_AUDIO_QUALITY = "LOW_AUDIO_QUALITY"
    TARGET_LOST = "TARGET_LOST"
    INSUFFICIENT_CONFIDENCE = "INSUFFICIENT_CONFIDENCE"
    INSUFFICIENT_ENROLLMENT_DATA = "INSUFFICIENT_ENROLLMENT_DATA"


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0


@dataclass
class ClothingAttributes:
    shirt_color: Optional[str] = None
    shirt_type: Optional[str] = None
    pants_color: Optional[str] = None
    gender_hint: Optional[str] = None
    raw_attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FaceTrack:
    face_id: str
    bbox: BoundingBox
    confidence: float
    embedding: Optional[np.ndarray] = None


@dataclass
class PersonTrack:
    person_id: str
    bounding_box: BoundingBox
    frame_idx: int
    timestamp_sec: float
    tracking_confidence: float
    face_available: bool = False
    body_visible: bool = True
    clothing_attributes: ClothingAttributes = field(default_factory=ClothingAttributes)
    position: str = "center"
    speaking_probability: float = 0.0
    speaker_embedding: Optional[np.ndarray] = None
    associated_face: Optional[FaceTrack] = None


@dataclass
class TargetQuery:
    raw_text: str
    structured_attributes: Dict[str, Any] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)


@dataclass
class CandidateScore:
    person_id: str
    visual_match_score: float
    tracking_score: float
    face_match_score: float
    position_score: float
    overall_target_score: float


@dataclass
class TargetSelectionResult:
    target_person_id: Optional[str]
    confidence: float
    status: TargetStatus
    candidates: List[CandidateScore] = field(default_factory=list)
    failure_reason: Optional[FailureReason] = None


@dataclass
class SpeechSegment:
    start_sec: float
    end_sec: float
    confidence: float = 1.0


@dataclass
class TranscriptSegment:
    speaker: str
    start: float
    end: float
    text: str
    confidence: float


@dataclass
class PipelineResult:
    target_person_id: Optional[str]
    target_audio_path: Optional[str] = None
    target_audio_enhanced_path: Optional[str] = None
    enrollment_path_used: Optional[EnrollmentPath] = None
    transcription: List[TranscriptSegment] = field(default_factory=list)
    timeline: Dict[str, Any] = field(default_factory=dict)
    overall_confidence: float = 0.0
    status: TargetStatus = TargetStatus.CONFIRMED
    failure_reason: Optional[FailureReason] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
