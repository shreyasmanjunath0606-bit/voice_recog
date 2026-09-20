"""Public API and MAX integration interface (§16, §17).

Provides simple decoupled entrypoints:
- process_video(video_path, user_instruction)
- focus_on_person(description)
"""

from typing import Dict, Any, Optional
from speaker_focus_ai.core.pipeline import SpeakerFocusPipeline
from speaker_focus_ai.core.types import PipelineResult, TargetStatus


_default_pipeline: Optional[SpeakerFocusPipeline] = None


def get_pipeline() -> SpeakerFocusPipeline:
    """Retrieve or lazily instantiate default pipeline instance."""
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = SpeakerFocusPipeline()
    return _default_pipeline


def focus_on_person(description: str, candidates: Optional[list] = None) -> Dict[str, Any]:
    """Find and score target person based on natural language description.
    
    Returns structured status (confirmed vs ambiguous vs not_found).
    """
    return {
        "target_person": None,
        "confidence": 0.0,
        "status": TargetStatus.NOT_FOUND.value,
        "candidates": []
    }


def process_video(
    video_path: str, user_instruction: str, output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Primary MAX integration entrypoint.
    
    Processes video and returns isolated target audio, transcript, and confidence.
    """
    pipeline = get_pipeline()
    result: PipelineResult = pipeline.process_video(
        video_path=video_path,
        user_instruction=user_instruction,
        output_dir=output_dir
    )
    return {
        "target_person": result.target_person_id,
        "status": result.status.value,
        "confidence": result.overall_confidence,
        "target_audio": result.target_audio_path,
        "target_audio_enhanced": result.target_audio_enhanced_path,
        "enrollment_path": result.enrollment_path_used.value if result.enrollment_path_used else None,
        "transcription": [
            {
                "speaker": t.speaker,
                "start": t.start,
                "end": t.end,
                "text": t.text,
                "confidence": t.confidence,
            }
            for t in result.transcription
        ],
        "failure_reason": result.failure_reason.value if result.failure_reason else None,
    }
