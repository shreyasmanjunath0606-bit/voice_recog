"""Main pipeline orchestrator coordinating vision, audio, multimodal, and speech modules."""

import logging
from typing import Optional

from speaker_focus_ai.core.config import PipelineConfig
from speaker_focus_ai.core.types import PipelineResult, TargetStatus

logger = logging.getLogger(__name__)


class SpeakerFocusPipeline:
    """End-to-end multimodal pipeline for target speaker isolation."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        # Modules will be initialized lazily or injected
        self._initialized = False

    def process_video(
        self,
        video_path: str,
        user_instruction: str,
        output_dir: Optional[str] = None
    ) -> PipelineResult:
        """Process video to isolate and transcribe speech of the described person.
        
        Args:
            video_path: Path to the input video file.
            user_instruction: Natural language description of the target person.
            output_dir: Directory where audio and transcripts should be saved.
            
        Returns:
            PipelineResult containing audio paths, transcription, timeline, confidence, status.
        """
        logger.info(f"Starting pipeline for video '{video_path}' with instruction '{user_instruction}'")
        
        # Initialize Multimodal Components
        from speaker_focus_ai.multimodal.active_speaker import ActiveSpeakerDetector
        from speaker_focus_ai.multimodal.confidence import ConfidenceEngine
        from speaker_focus_ai.debug.visualizer import DebugVisualizer
        
        # New imports for actual models
        from speaker_focus_ai.vision.person_detector import read_video_frames
        from speaker_focus_ai.vision.tracker import MultiPersonTracker
        from speaker_focus_ai.multimodal.target_matcher import TargetMatcher
        from speaker_focus_ai.audio.extractor import AudioExtractor
        from speaker_focus_ai.audio.separator import AudioSeparator
        from speaker_focus_ai.speech.transcription import SpeechTranscriber
        from speaker_focus_ai.core.types import FailureReason
        import os
        import numpy as np

        asd = ActiveSpeakerDetector()
        confidence_engine = ConfidenceEngine()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            debug_path = os.path.join(output_dir, "debug.mp4")
            visualizer = DebugVisualizer(output_video_path=debug_path)
        else:
            visualizer = DebugVisualizer()

        # Step 1: Video reading and Vision tracking
        tracker = MultiPersonTracker()
        for frame_idx, timestamp_sec, frame in read_video_frames(video_path):
            tracker.track_frame(frame, frame_idx, timestamp_sec)
        
        candidates = list(tracker.active_tracks.values())

        # Step 2: Target Matching
        matcher = TargetMatcher()
        query = matcher.parse_description(user_instruction)
        target_result = matcher.score_candidates(query, candidates)

        target_person_id = target_result.target_person_id
        if not target_person_id:
            return PipelineResult(
                target_person_id=None,
                status=TargetStatus.NOT_FOUND,
                failure_reason=target_result.failure_reason or FailureReason.TARGET_LOST,
                overall_confidence=0.0
            )

        target_track = next((t for t in candidates if t.person_id == target_person_id), None)
        face_tracks = target_track.face_sequence if target_track else []

        # Step 3: Audio Extraction
        extractor = AudioExtractor()
        try:
            audio_waveform, sample_rate = extractor.extract_waveform(video_path)
        except Exception as e:
            return PipelineResult(
                target_person_id=target_person_id,
                status=TargetStatus.NOT_FOUND,
                failure_reason=FailureReason.AUDIO_VIDEO_DESYNC,
                overall_confidence=0.0
            )

        # Step 4: Active Speaker Detection
        target_asd_prob = asd.compute_speaking_probability(face_tracks, audio_waveform, sample_rate=sample_rate)

        # Step 5: Audio Separation
        separator = AudioSeparator()
        target_waveform, sep_confidence = separator.separate_target(audio_waveform, sample_rate=sample_rate)
        
        target_audio_path = None
        if output_dir:
            target_audio_path = os.path.join(output_dir, "target_isolated.wav")
            extractor.save_wav(target_waveform, sample_rate, target_audio_path)

        # Step 6: Transcription
        transcriber = SpeechTranscriber()
        transcription = transcriber.transcribe(target_waveform, sample_rate=sample_rate, speaker_id=target_person_id)

        # Step 7: Confidence estimation
        metrics = {
            "face_visible_duration": len(face_tracks) / 25.0,
            "top_candidates_score_diff": 0.5,
            "total_speech_segments": len(transcription),
            "target_asd_prob": target_asd_prob,
            "separation_metric": sep_confidence,
            "has_enrollment_data": False
        }
        
        failure = confidence_engine.diagnose_failure(**metrics)
        if failure:
            return PipelineResult(
                target_person_id=target_person_id,
                status=TargetStatus.NOT_FOUND,
                failure_reason=failure,
                overall_confidence=0.0
            )
            
        overall_confidence = confidence_engine.calculate_confidence(
            visual_match=target_result.confidence, 
            speaker_match=0.5,
            asd_speaking_prob=target_asd_prob,
            tracking_stability=target_track.tracking_confidence if target_track else 0.5, 
            enrollment_confidence=0.0, 
            separation_quality=sep_confidence
        )
        
        # Clean up
        visualizer.release()
        
        return PipelineResult(
            target_person_id=target_person_id,
            status=TargetStatus.CONFIRMED,
            overall_confidence=overall_confidence,
            target_audio=target_audio_path,
            transcription=transcription
        )
