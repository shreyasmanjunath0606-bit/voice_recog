"""Global configuration for models, thresholds, weights, and hardware backend."""

from dataclasses import dataclass, field
from typing import Dict, Any


def get_best_device() -> str:
    """Automatically choose the best available hardware accelerator."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


@dataclass
class HardwareConfig:
    device: str = field(default_factory=get_best_device)  # Auto-selects cuda -> mps -> cpu
    fallback_to_cpu: bool = True
    fp16: bool = True
    lazy_loading: bool = True


@dataclass
class MatchingWeights:
    shirt_color: float = 0.35
    shirt_type: float = 0.15
    pants_color: float = 0.20
    position: float = 0.10
    gender_hint: float = 0.05
    face_info: float = 0.15


@dataclass
class ConfidenceWeights:
    visual_match: float = 0.25
    speaker_match: float = 0.20
    asd_speaking_prob: float = 0.25
    tracking_stability: float = 0.15
    enrollment_confidence: float = 0.15


@dataclass
class PipelineConfig:
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    matching_weights: MatchingWeights = field(default_factory=MatchingWeights)
    confidence_weights: ConfidenceWeights = field(default_factory=ConfidenceWeights)
    ambiguity_threshold: float = 0.08  # if diff between top 2 candidates <= threshold, mark AMBIGUOUS
    min_confidence_threshold: float = 0.60
    min_enrollment_duration_sec: float = 1.5
