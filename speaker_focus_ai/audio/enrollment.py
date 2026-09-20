"""Voiceprint enrollment & bootstrapping (§10).

Handles both preferred path (clean solo segments from ASD) and fallback path
(blind source separation + provisional voiceprint extraction when target never speaks alone).
"""

from typing import List, Optional, Tuple, Dict
import numpy as np

from speaker_focus_ai.core.types import EnrollmentPath, FailureReason
from speaker_focus_ai.audio.speaker_embedding import SpeakerEmbeddingExtractor


class VoiceprintEnrollmentManager:
    """Manages clean voiceprint enrollment and provisional bootstrapping."""

    def __init__(
        self,
        embedding_extractor: Optional[SpeakerEmbeddingExtractor] = None,
        min_solo_duration_sec: float = 1.5,
        prob_high: float = 0.85,
        prob_low: float = 0.15
    ):
        self.min_solo_duration_sec = min_solo_duration_sec
        self.prob_high = prob_high
        self.prob_low = prob_low
        self.embedding_extractor = embedding_extractor or SpeakerEmbeddingExtractor()
        
        self.enrolled_voiceprints: Dict[str, np.ndarray] = {}
        self.enrollment_paths: Dict[str, EnrollmentPath] = {}

    def find_clean_segments(
        self,
        person_id: str,
        asd_timelines: Dict[str, List[Tuple[float, float, float]]]
    ) -> List[Tuple[float, float]]:
        """Identify time windows where only target person_id has high speaking probability (P > 0.85)
        and all other speakers are silent (P < 0.15).
        
        Args:
            person_id: ID of the target speaker.
            asd_timelines: Dict mapping person_id to list of (start_sec, end_sec, speaking_prob).
            
        Returns:
            List of clean (start_sec, end_sec) intervals where person speaks alone.
        """
        if person_id not in asd_timelines:
            return []

        target_intervals = asd_timelines[person_id]
        other_person_ids = [pid for pid in asd_timelines.keys() if pid != person_id]

        clean_segments = []

        for start_sec, end_sec, target_prob in target_intervals:
            if target_prob < self.prob_high:
                continue

            # Check if any other speaker is active during this interval
            is_solo = True
            for other_id in other_person_ids:
                for o_start, o_end, o_prob in asd_timelines[other_id]:
                    # Check temporal overlap
                    if max(start_sec, o_start) < min(end_sec, o_end):
                        if o_prob >= self.prob_low:
                            is_solo = False
                            break
                if not is_solo:
                    break

            if is_solo:
                clean_segments.append((start_sec, end_sec))

        return clean_segments

    def enroll_clean(
        self,
        person_id: str,
        audio_waveform: np.ndarray,
        clean_segments: List[Tuple[float, float]],
        sample_rate: int = 16000
    ) -> Tuple[np.ndarray, float]:
        """Enroll high-confidence voiceprint by averaging embeddings over clean solo segments.
        
        Args:
            person_id: Target speaker ID.
            audio_waveform: Full audio waveform array.
            clean_segments: List of (start_sec, end_sec) clean solo segments.
            sample_rate: Audio sampling rate.
            
        Returns:
            Tuple of (192-dim normalized embedding array, confidence_score).
        """
        total_duration = sum(end - start for start, end in clean_segments)
        if total_duration < self.min_solo_duration_sec:
            raise ValueError(
                f"Insufficient clean solo duration ({total_duration:.2f}s < required {self.min_solo_duration_sec}s)"
            )

        segment_embeddings = []
        for start_sec, end_sec in clean_segments:
            start_idx = int(start_sec * sample_rate)
            end_idx = int(end_sec * sample_rate)
            chunk = audio_waveform[start_idx:end_idx]

            if len(chunk) > 0:
                emb = self.embedding_extractor.compute_embedding(chunk, sample_rate=sample_rate)
                segment_embeddings.append(emb)

        if not segment_embeddings:
            raise RuntimeError("Failed to extract embeddings from clean segments.")

        # Average embeddings across clean segments
        avg_emb = np.mean(segment_embeddings, axis=0)
        norm = np.linalg.norm(avg_emb)
        if norm > 0:
            avg_emb = avg_emb / norm

        confidence = min(0.99, 0.85 + 0.05 * (total_duration / self.min_solo_duration_sec))
        
        self.enrolled_voiceprints[person_id] = avg_emb
        self.enrollment_paths[person_id] = EnrollmentPath.CLEAN

        return avg_emb, float(confidence)

    def bootstrap_provisional(
        self,
        person_id: str,
        audio_waveform: np.ndarray,
        separated_streams: List[np.ndarray],
        asd_timeline: List[Tuple[float, float, float]],
        sample_rate: int = 16000
    ) -> Tuple[np.ndarray, float]:
        """Fallback path: Extract provisional voiceprint from separated stream best matching ASD timing.
        
        Args:
            person_id: Target speaker ID.
            audio_waveform: Full mixed audio waveform.
            separated_streams: List of separated single-speaker audio arrays.
            asd_timeline: Target speaker ASD probability timeline (start_sec, end_sec, prob).
            sample_rate: Audio sampling rate.
            
        Returns:
            Tuple of (provisional 192-dim embedding, correlation_score).
        """
        if not separated_streams or not asd_timeline:
            self.enrollment_paths[person_id] = EnrollmentPath.FAILED
            raise RuntimeError(FailureReason.INSUFFICIENT_ENROLLMENT_DATA.value)

        # Compute energy envelopes for each separated stream over frame intervals
        best_stream_idx = -1
        max_correlation = -1.0

        for idx, stream in enumerate(separated_streams):
            if len(stream) == 0:
                continue

            stream_energies = []
            target_probs = []

            for start_sec, end_sec, prob in asd_timeline:
                start_idx = int(start_sec * sample_rate)
                end_idx = int(end_sec * sample_rate)
                
                chunk = stream[start_idx:end_idx] if start_idx < len(stream) else np.array([])
                energy = float(np.sqrt(np.mean(chunk**2))) if len(chunk) > 0 else 0.0

                stream_energies.append(energy)
                target_probs.append(prob)

            if len(stream_energies) > 1 and np.std(stream_energies) > 1e-6:
                # Compute Pearson correlation between audio energy and ASD mouth movement curve
                correlation = float(np.corrcoef(stream_energies, target_probs)[0, 1])
                if np.isnan(correlation):
                    correlation = 0.0
            else:
                correlation = 0.0

            if correlation > max_correlation:
                max_correlation = correlation
                best_stream_idx = idx

        # Quality threshold check for provisional bootstrapping
        if best_stream_idx < 0 or max_correlation < 0.20:
            self.enrollment_paths[person_id] = EnrollmentPath.FAILED
            raise RuntimeError(FailureReason.INSUFFICIENT_ENROLLMENT_DATA.value)

        best_stream = separated_streams[best_stream_idx]
        provisional_emb = self.embedding_extractor.compute_embedding(best_stream, sample_rate=sample_rate)

        confidence = max(0.50, min(0.75, max_correlation))
        
        self.enrolled_voiceprints[person_id] = provisional_emb
        self.enrollment_paths[person_id] = EnrollmentPath.PROVISIONAL

        return provisional_emb, float(confidence)

    def enroll_target(
        self,
        person_id: str,
        audio_waveform: np.ndarray,
        asd_timelines: Dict[str, List[Tuple[float, float, float]]],
        separated_streams_fallback: Optional[List[np.ndarray]] = None,
        sample_rate: int = 16000
    ) -> Tuple[np.ndarray, EnrollmentPath, float]:
        """Main enrollment pipeline entry point attempting clean path first, then provisional fallback.
        
        Returns:
            Tuple of (embedding, enrollment_path, confidence_score).
        """
        clean_segments = self.find_clean_segments(person_id, asd_timelines)
        total_clean_duration = sum(end - start for start, end in clean_segments)

        # Path A: Clean Solo Enrollment
        if total_clean_duration >= self.min_solo_duration_sec:
            emb, conf = self.enroll_clean(person_id, audio_waveform, clean_segments, sample_rate=sample_rate)
            return emb, EnrollmentPath.CLEAN, conf

        # Path B: Provisional Fallback Enrollment
        if separated_streams_fallback and person_id in asd_timelines:
            target_asd = asd_timelines[person_id]
            try:
                emb, conf = self.bootstrap_provisional(
                    person_id, audio_waveform, separated_streams_fallback, target_asd, sample_rate=sample_rate
                )
                return emb, EnrollmentPath.PROVISIONAL, conf
            except Exception:
                pass

        # Failure Path
        self.enrollment_paths[person_id] = EnrollmentPath.FAILED
        raise RuntimeError(FailureReason.INSUFFICIENT_ENROLLMENT_DATA.value)
