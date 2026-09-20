# MAX — Speaker Focus AI

Target Speaker Isolation & Visual-Audio Association AI module for MAX (Jarvis private assistant).

## Architecture Overview

```text
speaker_focus_ai/
│
├── core/
│   ├── pipeline.py            # Main pipeline orchestrator
│   ├── config.py              # Configuration & hardware backends
│   └── types.py               # Shared data types & schemas
│
├── vision/
│   ├── person_detector.py     # Body detection
│   ├── face_detector.py       # Face detection & association
│   ├── tracker.py             # Multi-person tracking (ByteTrack)
│   └── attributes.py          # Clothing & spatial attributes
│
├── audio/
│   ├── extractor.py           # Video audio extraction
│   ├── vad.py                 # Voice Activity Detection (Silero)
│   ├── speaker_embedding.py   # Speaker embeddings (ECAPA-TDNN)
│   ├── enrollment.py          # Clean & provisional voiceprint bootstrapping (§10)
│   ├── separator.py           # Neural source separation
│   └── enhancer.py            # Speech enhancement & denoising
│
├── multimodal/
│   ├── target_matcher.py      # Natural language description matching
│   ├── active_speaker.py      # Audio-Visual Active Speaker Detection (ASD)
│   └── confidence.py          # Multimodal confidence scoring engine
│
├── speech/
│   └── transcription.py       # Local timestamped transcription (Whisper)
│
├── interface/
│   └── api.py                 # MAX integration API
│
├── debug/
│   └── visualizer.py          # Video overlay telemetry visualizer
│
├── tests/                     # Unit and integration test suite
│
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Run test suite
pytest speaker_focus_ai/tests/
```
