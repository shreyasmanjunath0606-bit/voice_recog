"""Local speech transcription with timestamps using faster-whisper and PyTorch Whisper fallback."""

import os
from typing import List, Optional, Union
import numpy as np
import torch

from speaker_focus_ai.core.types import TranscriptSegment


class SpeechTranscriber:
    """Transcribes audio with word/segment level timestamps."""

    def __init__(
        self,
        model_size: str = "tiny.en",
        device: Optional[str] = None,
        compute_type: Optional[str] = None
    ):
        self.model_size = model_size

        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        if compute_type is None:
            self.compute_type = "float16" if self.device == "cuda" else "int8"
        else:
            self.compute_type = compute_type

        self._model = None
        self._engine = None  # "faster-whisper" or "openai-whisper"

    def load_model(self) -> None:
        """Lazily load Whisper transcription model."""
        if self._model is not None:
            return

        # Attempt 1: faster-whisper (CTranslate2)
        try:
            from faster_whisper import WhisperModel
            device_str = "cuda" if self.device == "cuda" else "cpu"
            self._model = WhisperModel(self.model_size, device=device_str, compute_type=self.compute_type)
            self._engine = "faster-whisper"
            return
        except Exception as e:
            # Fallback for Windows AppLocker policy or missing CTranslate2 DLL
            pass

        # Attempt 2: PyTorch native openai-whisper
        try:
            import whisper
            device_target = torch.device(self.device if self.device != "mps" else "cpu")
            self._model = whisper.load_model(self.model_size, device=device_target)
            self._engine = "openai-whisper"
            return
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model '{self.model_size}': {e}")

    def transcribe(
        self,
        audio_input: Union[np.ndarray, str],
        sample_rate: int = 16000,
        speaker_id: str = "target"
    ) -> List[TranscriptSegment]:
        """Transcribe audio waveform or audio file path into timestamped segments.
        
        Args:
            audio_input: Audio numpy array or absolute path to WAV file.
            sample_rate: Audio sampling rate (default 16000).
            speaker_id: Target speaker ID for attribution.
            
        Returns:
            List of TranscriptSegment dataclasses.
        """
        self.load_model()

        if isinstance(audio_input, np.ndarray):
            if len(audio_input) == 0:
                return []
            audio_data = audio_input.astype(np.float32)
        elif isinstance(audio_input, str):
            if not os.path.exists(audio_input):
                raise FileNotFoundError(f"Audio file not found: {audio_input}")
            audio_data = audio_input
        else:
            return []

        segments_out: List[TranscriptSegment] = []

        if self._engine == "faster-whisper":
            segments, info = self._model.transcribe(
                audio_data,
                beam_size=5,
                language="en" if self.model_size.endswith(".en") else None
            )

            for seg in segments:
                text = seg.text.strip()
                if text:
                    # Convert logprob to confidence [0.0, 1.0]
                    confidence = float(np.exp(min(0.0, seg.avg_logprob)))
                    segments_out.append(
                        TranscriptSegment(
                            speaker=speaker_id,
                            start=float(seg.start),
                            end=float(seg.end),
                            text=text,
                            confidence=confidence
                        )
                    )

        elif self._engine == "openai-whisper":
            res = self._model.transcribe(
                audio_data,
                fp16=(self.device == "cuda")
            )

            for seg in res.get("segments", []):
                text = seg.get("text", "").strip()
                if text:
                    avg_logprob = seg.get("avg_logprob", 0.0)
                    confidence = float(np.exp(min(0.0, avg_logprob)))
                    segments_out.append(
                        TranscriptSegment(
                            speaker=speaker_id,
                            start=float(seg["start"]),
                            end=float(seg["end"]),
                            text=text,
                            confidence=confidence
                        )
                    )

        return segments_out
