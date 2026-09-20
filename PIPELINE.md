# Execution Pipeline & Phase Implementation Guide (`PIPELINE.md`)

## 1. Development & Phased Implementation Plan

Per project specifications (§26, §29), modules are developed incrementally. Tests must verify each phase before moving forward.

```
PHASE 0: Hardware Smoke Test (MPS acceleration & CPU fallback verification)
   │
PHASE 1: Video loading & frame extraction (FFmpeg / OpenCV / PyAV)
   │
PHASE 2: Person detection & multi-person tracking (YOLO + ByteTrack)
   │
PHASE 3: Face detection & person/face association
   │
PHASE 4: Visual attribute extraction (clothing, colors, spatial position)
   │
PHASE 5: Natural-language target candidate matching & scoring
   │
PHASE 6: Audio extraction & Voice Activity Detection (Silero VAD)
   │
PHASE 7: Speaker embedding generation (ECAPA-TDNN)
   │
PHASE 8: Audio-Visual Active Speaker Detection (TalkNet / Light-ASD)
   │
PHASE 9: Voiceprint enrollment & bootstrapping (Clean solo vs. provisional fallback)
   │
PHASE 10: Target speaker source separation (Voiceprint-conditioned)
   │
PHASE 11: Speech transcription & word-level alignment (faster-whisper)
   │
PHASE 12: Confidence scoring engine & failure reason diagnosis
   │
PHASE 13: Telemetry visualizer & debug overlay video generation
   │
PHASE 14: MAX AI assistant integration API
   │
PHASE 15: Apple Silicon performance optimization & memory management
```

## 2. Phase 0 Hardware Smoke Test Specification

Before implementing downstream models:
1. Verify PyTorch sees Apple Silicon MPS (`torch.backends.mps.is_available()`).
2. Verify candidate models run forward passes without falling back to unimplemented CUDA operations.
3. Test CPU fallback behavior to ensure smooth local operation.

## 3. Failure State Handling Contract (§23)

| Failure State | Trigger Condition | System Action |
| :--- | :--- | :--- |
| `FACE_NOT_VISIBLE` | Candidate has no detectable face for > 3 seconds | Flags degraded visual confidence |
| `AMBIGUOUS_TARGET` | Top candidate scores difference $\le 0.08$ | Returns status `ambiguous` with candidate list |
| `NO_SPEECH_DETECTED`| VAD detects 0 speech segments in entire clip | Returns early without running separator |
| `TARGET_NOT_SPEAKING`| Target person tracked, but ASD speaking prob $< 0.15$ | Returns status `confirmed`, marks no target audio |
| `LOW_SEPARATION_CONFIDENCE` | Separation metric $< 0.35$ | Flags low separation quality, alerts user |
| `INSUFFICIENT_ENROLLMENT_DATA` | No clean solo segments and provisional stream fails | Returns failure reason explicitly |
