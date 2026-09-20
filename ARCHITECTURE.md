# MAX — Target Speaker Isolation Architecture (`ARCHITECTURE.md`)

## 1. System Architecture Overview

MAX Speaker Focus AI is a local, modular multimodal system designed to isolate and transcribe the speech of a specific target person in a multi-speaker video environment. The system integrates computer vision, audio processing, multimodal active speaker detection, and speech transcription.

```text
                               ┌────────────────────────┐
                               │      Input Video       │
                               └───────────┬────────────┘
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
        ┌───────────────────────┐                     ┌───────────────────────┐
        │     Vision Stream     │                     │     Audio Stream      │
        └───────────┬───────────┘                     └───────────┬───────────┘
                    │                                             │
      ┌─────────────┴─────────────┐                               │
      │ Person Detection & Track  │                               │
      │ Face Detection & Assoc    │                               │
      │ Visual Attributes         │                               │
      └─────────────┬─────────────┘                               │
                    │                                             │
                    │   Natural Language Description              │
                    │        (e.g., "guy in blue")                │
                    ▼                                             ▼
        ┌───────────────────────┐                     ┌───────────────────────┐
        │  Target Candidate     │                     │ Audio Extraction &    │
        │  Matching & Scoring   │                     │ Voice Activity Detect │
        └───────────┬───────────┘                     └───────────┬───────────┘
                    │                                             │
                    └──────────────────────┬──────────────────────┘
                                           ▼
                       ┌──────────────────────────────────────┐
                       │ Audio-Visual Active Speaker Detection│
                       │   (TalkNet-ASD / Light-ASD)          │
                       │   Correlates mouth motion with audio │
                       └───────────────────┬──────────────────┘
                                           ▼
                       ┌──────────────────────────────────────┐
                       │ Voiceprint Enrollment & Bootstrapping│
                       │ - Preferred: Solo segment enrollment │
                       │ - Fallback: Blind sep + ASD timing   │
                       └───────────────────┬──────────────────┘
                                           ▼
                       ┌──────────────────────────────────────┐
                       │ Speech Source Separation & Enhancer  │
                       │ Target voice extraction conditioned  │
                       │ on voiceprint; denoising             │
                       └───────────────────┬──────────────────┘
                                           ▼
                       ┌──────────────────────────────────────┐
                       │ Speech Transcription & Alignment     │
                       │ Timestamped transcript & confidence  │
                       └───────────────────┬──────────────────┘
                                           ▼
                       ┌──────────────────────────────────────┐
                       │   Confidence Engine & Debug Output   │
                       │ Transparent scoring & verification   │
                       └──────────────────────────────────────┘
```

## 2. Component Input / Output Specifications

| Component | Module | Input | Output |
| :--- | :--- | :--- | :--- |
| **Person Detector** | `vision.person_detector` | Video Frame `(H, W, 3)` | Bounding boxes `List[BoundingBox]` |
| **Face Detector** | `vision.face_detector` | Video Frame `(H, W, 3)` | Face tracks & embeddings `List[FaceTrack]` |
| **Multi-Person Tracker**| `vision.tracker` | Bounding boxes + frame | Stable identity tracks `List[PersonTrack]` |
| **Attribute Extractor** | `vision.attributes` | Person crop `(H, W, 3)` | `ClothingAttributes` (colors, position) |
| **Target Matcher** | `multimodal.target_matcher` | Natural language text + tracks | `TargetSelectionResult` (confirmed/ambiguous) |
| **Audio Extractor** | `audio.extractor` | Video file path | 16kHz mono PCM waveform `(N,)` |
| **VAD** | `audio.vad` | Audio waveform | `List[SpeechSegment]` (timestamps) |
| **Active Speaker Det** | `multimodal.active_speaker` | Face crop sequences + audio | Per-person speaking probability timeline |
| **Voiceprint Enrollment**| `audio.enrollment` | Audio + ASD timeline | Target voiceprint vector `(192,)` + status |
| **Source Separator** | `audio.separator` | Mixed audio + voiceprint | Isolated target waveform `(N,)` + metric |
| **Speech Enhancer** | `audio.enhancer` | Separated waveform | Enhanced waveform `(N,)` |
| **Transcriber** | `speech.transcription` | Target waveform | `List[TranscriptSegment]` (text, timestamps) |
| **Confidence Engine** | `multimodal.confidence` | All sub-scores | Composite confidence `float [0.0, 1.0]` |

## 3. Communication & Data Flow

Modules communicate using strongly-typed dataclasses defined in `speaker_focus_ai/core/types.py`.
- Vision metadata and audio streams are processed in temporal synchronization.
- Timestamps in seconds (`float`) serve as the universal temporal key across tracks, audio windows, and transcripts.

## 4. Known Simplifications & Limitations (v1)

1. **Confidence Score Independence Assumption (§22):** The initial v1 scoring engine combines signals via weighted sum. While `speaker_match` and `asd_speaking_probability` exhibit correlation in degraded conditions, the weighted sum serves as a robust baseline. A learned combiner (e.g. logistic calibration) will be trained once test datasets (§27) are evaluated.
2. **Ambiguity Handling:** When top candidate match scores differ by $\le 0.08$, the system returns `AMBIGUOUS_TARGET` to prevent silent misidentification.
