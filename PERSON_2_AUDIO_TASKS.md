# 🧑‍💻 PERSON 2: Audio, Voiceprint Enrollment & Speech Separation Lead

**Title:** Audio Processing & Separation Lead ("The Ears")  
**Git Branch:** `feat/audio-separation`  
**Base Folder:** `speaker_focus_ai/audio/` & `speaker_focus_ai/speech/`

---

## 🎯 Primary Mission
Extract audio from the video, detect speech activity (VAD), enroll speaker voiceprints (both clean solo segments and provisional fallback per §10), isolate the target speaker's voice from overlapping talkers using neural source separation, and transcribe the target's speech with timestamps.

---

## 📂 Owned Files & Responsibilities

| File Path | Description | Spec Section |
| :--- | :--- | :--- |
| `speaker_focus_ai/audio/extractor.py` | Extract 16kHz mono WAV from video container | §9 |
| `speaker_focus_ai/audio/vad.py` | Voice Activity Detection using Silero VAD | §9 |
| `speaker_focus_ai/audio/speaker_embedding.py`| Extract 192-dim speaker vectors (ECAPA-TDNN / SpeechBrain) | §9 |
| `speaker_focus_ai/audio/enrollment.py` | Voiceprint enrollment manager: clean vs. provisional bootstrap | §10 |
| `speaker_focus_ai/audio/separator.py` | Target speaker separation (SepFormer / Conv-TasNet / SpEx+) | §11, §12 |
| `speaker_focus_ai/audio/enhancer.py` | Speech denoising & enhancement (DeepFilterNet) | §12 |
| `speaker_focus_ai/speech/transcription.py` | Local transcription with timestamps (faster-whisper) | §13 |
| `speaker_focus_ai/tests/test_audio.py` | Unit and integration tests for audio & separation | §26 |

---

## 📋 Step-by-Step Task Checklist

### Milestone 1: Phase 0 (Audio) — Hardware Smoke Test
- [x] Test SpeechBrain (ECAPA-TDNN) and faster-whisper on Apple Silicon.
- [x] Verify if SepFormer / Conv-TasNet runs on MPS or if it cleanly falls back to CPU.
- [x] Record hardware execution results in `MODEL_SELECTION.md`.

### Milestone 2: Phase 6 — Audio Extraction & VAD
- [x] In `extractor.py`, implement `extract_waveform` using PyAV, `ffmpeg`, or `librosa`.
- [x] Ensure all audio is converted to **16,000 Hz, mono channel, float32 [-1.0, 1.0]**.
- [x] In `vad.py`, integrate **Silero VAD** (`torch.hub.load('snakers4/silero-vad')`).
- [x] Return list of `SpeechSegment(start_sec=..., end_sec=..., confidence=...)`.
- [x] Flag `NO_SPEECH_DETECTED` if VAD finds zero speech in the entire file.

### Milestone 3: Phase 7 — Speaker Representation
- [x] In `speaker_embedding.py`, load **ECAPA-TDNN** (e.g., `speechbrain/spkrec-ecapa-voxceleb`).
- [x] Compute 192-dimensional L2-normalized voice vector for any given audio segment.
- [x] Implement cosine similarity helper `compute_similarity(emb1, emb2)`.

### Milestone 4: Phase 9 — Voiceprint Enrollment & Bootstrapping (§10)
- [x] In `enrollment.py`, implement **Preferred Path (Clean Solo Enrollment):**
  - Read the ASD timeline (from Person 3) to find intervals where only the target person has high speaking probability ($P > 0.85$) and others are silent ($P < 0.15$).
  - If total solo duration $\ge 1.5$ seconds, extract & average embeddings $\rightarrow$ mark `EnrollmentPath.CLEAN`.
- [x] Implement **Fallback Path (Provisional Bootstrap):**
  - If the target never speaks alone across the entire video:
  - Run blind source separation (milestone 5) on the overlapping section where the target is active.
  - Correlate each separated stream's energy profile with the target's mouth-movement timeline (from Person 3) to pick the best stream.
  - Extract provisional voiceprint $\rightarrow$ mark `EnrollmentPath.PROVISIONAL`.
  - If separation quality is too poor, raise `FailureReason.INSUFFICIENT_ENROLLMENT_DATA`.

### Milestone 5: Phase 10 — Target Speaker Separation & Enhancement
- [x] In `separator.py`, implement voiceprint-conditioned separation (SepFormer or SpEx+).
- [x] Extract target audio stream: `target_audio.wav`.
- [x] In `enhancer.py`, implement light noise suppression (DeepFilterNet) without distorting natural speech: `target_audio_enhanced.wav`.
- [x] Save both unenhanced and enhanced files.

### Milestone 6: Phase 11 — Speech Transcription
- [x] In `transcription.py`, integrate **faster-whisper** (`model_size="base.en"` or `"small.en"`).
- [x] Transcribe `target_audio.wav`.
- [x] Output list of `TranscriptSegment(speaker=target_id, start=..., end=..., text=..., confidence=...)`.

---

## 🤝 Cross-Team Integration Handoffs

1. **Intake from Person 3 (Active Speaker Detection):**
   * Receive the per-person speaking timeline `asd_timelines` (start, end, prob) to know when the target speaks and find clean solo segments.
2. **Handoff to Person 3 (Pipeline Orchestrator):**
   * Provide the separated WAV paths (`target_audio.wav`, `target_audio_enhanced.wav`), `enrollment_path_used`, and `transcription` list.

---

## 🧪 Verification & Testing
Run your tests locally:
```bash
python3 -m unittest speaker_focus_ai/tests/test_audio.py
```
* Test VAD on audio containing silence vs. active speech.
* Test voiceprint cosine similarity between two clips of the same speaker (> 0.70) vs. different speakers (< 0.40).
* Verify transcription produces accurate timestamps.
