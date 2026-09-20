# Model Selection & Hardware Compatibility (`MODEL_SELECTION.md`)

Evaluation of candidate open-source models for Apple Silicon Mac (Metal Performance Shaders / MPS and CPU fallback).

## Summary Matrix

| Task | Primary Candidate | Backup / Lightweight | Apple Silicon (MPS / CPU) | License |
| :--- | :--- | :--- | :--- | :--- |
| **Person Detection** | YOLOv8-nano / YOLOv8-small | MediaPipe Pose / Object | MPS accelerated (PyTorch / ONNX) | AGPL-3.0 / Apache 2.0 |
| **Face Detection** | MediaPipe Face Detection | RetinaFace / InsightFace | CPU / MPS (Highly optimized) | Apache 2.0 / MIT |
| **Tracking** | ByteTrack | BoT-SORT | CPU (pure algorithmic + Kalman) | MIT |
| **Active Speaker Det (ASD)**| TalkNet-ASD / Light-ASD | ASD-Transformer | PyTorch MPS or clean CPU fallback | MIT / Apache 2.0 |
| **Voice Activity Detection**| Silero VAD v4/v5 | WebRTC VAD | PyTorch / ONNX (ultra-low latency) | MIT |
| **Speaker Embeddings** | ECAPA-TDNN (SpeechBrain) | Resemblyzer | PyTorch MPS / CPU compatible | Apache 2.0 / MIT |
| **Source Separation** | SepFormer / Conv-TasNet | VoiceFilter-lite / SpEx+ | PyTorch CPU / MPS (Verify in Phase 0) | Apache 2.0 / MIT |
| **Speech Enhancement** | DeepFilterNet | VoiceFixer / DTLN | Rust/C++ CPU / PyTorch | MIT |
| **Speech-to-Text** | faster-whisper (tiny/base) | Whisper.cpp | Apple Silicon NEON / Metal optimized | MIT |

## Detailed Model Evaluations

### 1. Person Detection: YOLOv8n
- **Input:** RGB image `(640, 640, 3)`
- **Output:** Bounding boxes `(x1, y1, x2, y2, conf, class_id)`
- **Model Size:** ~6.3 MB
- **Hardware:** Runs natively on MPS (`device='mps'`) and achieves 60+ FPS on Apple Silicon.

### 2. Audio-Visual Active Speaker Detection: TalkNet-ASD
- **Purpose:** Jointly process face crops and audio spectrogram to output speaking probability $P(\text{speaking})$.
- **Input:** Sequence of face crops `(T, 112, 112, 1)` + Audio Mel Spectrogram `(T, F)`.
- **Output:** Frame-level probability vector `(T,)`.
- **Hardware:** Requires verification in Phase 0 hardware smoke test.

### 3. Voiceprint Enrollment: ECAPA-TDNN
- **Purpose:** Generate 192-dimensional L2-normalized voice vector for target speaker matching and separation conditioning.
- **Input:** 16kHz mono audio segment (>1.5s).
- **Output:** `(192,)` embedding vector.
- **Hardware:** Runs cleanly on CPU and MPS via SpeechBrain.

### 4. Speech Source Separation: SepFormer / Conv-TasNet
- **Purpose:** Separate target speaker from overlapping multi-speaker mixture.
- **Input:** 16kHz mixed waveform + target conditioning voiceprint.
- **Hardware Note:** Speech separation models often have custom CUDA kernels; must test on MPS/CPU during Phase 0.

### 5. Speech-to-Text: faster-whisper
- **Purpose:** Generate accurate words and segment timestamps.
- **Model:** `base.en` or `small.en` via CTranslate2.
- **Hardware:** Optimized for ARM64 CPU with float32/int8 quantization.
