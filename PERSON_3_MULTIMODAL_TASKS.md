# 🧑‍💻 PERSON 3: Multimodal ASD, Pipeline Integration & Evaluation Lead

**Title:** Multimodal Fusion, System Pipeline & Evaluation Lead ("The Brain & Glue")  
**Git Branch:** `feat/multimodal-pipeline`  
**Base Folder:** `speaker_focus_ai/multimodal/`, `speaker_focus_ai/core/`, `speaker_focus_ai/debug/`, `speaker_focus_ai/interface/`

---

## 🎯 Primary Mission
Correlate mouth movement with audio using Audio-Visual Active Speaker Detection (ASD), fuse the outputs of Person 1 (Vision) and Person 2 (Audio) into the unified pipeline, calculate un-fabricated confidence scores, create the debug video overlay visualizer, and build the test suite (§27).

---

## 📂 Owned Files & Responsibilities

| File Path | Description | Spec Section |
| :--- | :--- | :--- |
| `speaker_focus_ai/multimodal/active_speaker.py` | Audio-Visual Active Speaker Detection (TalkNet / Light-ASD) | §8 |
| `speaker_focus_ai/multimodal/confidence.py` | Transparent multi-signal confidence calculation engine | §22 |
| `speaker_focus_ai/core/pipeline.py` | Master pipeline coordinating vision, audio, ASD, & separation | §3 |
| `speaker_focus_ai/debug/visualizer.py` | Diagnostic video visualizer with telemetry HUD overlay | §15 |
| `speaker_focus_ai/interface/api.py` | Clean public API for MAX Jarvis assistant integration | §16, §17 |
| `speaker_focus_ai/tests/test_pipeline.py` | End-to-end integration tests & test video curation | §26, §27 |

---

## 📋 Step-by-Step Task Checklist

### Milestone 1: Phase 0 — Hardware Smoke Test Coordinator
- [ ] Write a standalone smoke test script `smoke_test.py` that verifies:
  - PyTorch sees Apple Silicon GPU acceleration: `torch.backends.mps.is_available()`.
  - OpenCV, PyAV, and TorchAudio can open sample media.
- [ ] Run this with Person 1 and Person 2 on their machines to ensure zero hardware environment blockers.

### Milestone 2: Phase 8 — Audio-Visual Active Speaker Detection (ASD) (§8)
- [ ] Implement `active_speaker.py` using **TalkNet-ASD** (or Light-ASD).
- [ ] Take face crops `(T, 112, 112)` from Person 1 and the audio spectrogram window from Person 2.
- [ ] Compute per-frame speaking probability $P(\text{speaking})$ for each tracked person.
- [ ] Distinguish talking vs. standing silently vs. non-speech mouth motion (chewing, laughing).
- [ ] Build the complete timeline of who is speaking across time windows (§14).

### Milestone 3: Phase 12 — Multimodal Confidence Engine & Failure Handling (§22, §23)
- [ ] In `confidence.py`, calculate final target confidence from measurable signals:
  - `visual_match` (from Person 1)
  - `speaker_match` (from Person 2)
  - `asd_speaking_probability` (from ASD)
  - `tracking_stability` (from Person 1)
  - `enrollment_confidence` (clean = 1.0, provisional = 0.6)
  - `separation_quality` penalty factor
- [ ] Implement graceful failure handling (§23):
  - `FACE_NOT_VISIBLE`
  - `AMBIGUOUS_TARGET`
  - `NO_SPEECH_DETECTED`
  - `TARGET_NOT_SPEAKING`
  - `LOW_SEPARATION_CONFIDENCE`
  - `INSUFFICIENT_ENROLLMENT_DATA`

### Milestone 4: Phase 13 — Debug Telemetry Visualizer (§15)
- [ ] In `visualizer.py`, render debug video overlay on input video frames:
  - Bounding box around each person with ID.
  - Face bounding box with speaking status (`SPEAKING 93%` vs `NOT SPEAKING 12%`).
  - Target badge (`TARGET ★`).
  - Lower third HUD showing telemetry (ASD prob, speaker match %, enrollment path, overall confidence).

### Milestone 5: Phase 14 — Pipeline Integration & MAX API (§3, §16, §17)
- [ ] Connect all modules in `core/pipeline.py`:
  1. Video frame extraction & Audio extraction.
  2. Person & face tracking (Person 1).
  3. Query interpretation & Target matching (Person 1).
  4. Active Speaker Detection on all face tracks.
  5. Voiceprint enrollment (Person 2).
  6. Source separation & enhancement (Person 2).
  7. Speech transcription (Person 2).
  8. Composite confidence estimation & debug video generation.
- [ ] Expose clean API in `interface/api.py`:
  - `process_video(video_path, user_instruction)`
  - `focus_on_person(description)`

### Milestone 6: Phase 15 & Test Video Curation (§27)
- [ ] Curate or generate controlled test clips for the conditions specified in §27:
  - 1 person speaking, 2 people speaking sequentially, 2+ people overlapping.
  - **Audio/video desync test case:** intentionally desynced audio to verify ASD failure detection.

---

## 🤝 Cross-Team Integration Handoffs

1. **Intake from Person 1:**
   * Receive `List[PersonTrack]`, face bounding box crops, and confirmed `target_person_id`.
2. **Intake from Person 2:**
   * Receive raw audio waveform, VAD segments, voiceprint vectors, and separated audio files.
3. **Delivery to Person 2:**
   * Send the computed ASD timeline to Person 2 so they can identify clean solo segments for voiceprint enrollment.

---

## 🧪 Verification & Testing
Run your tests locally:
```bash
python3 -m unittest speaker_focus_ai/tests/test_pipeline.py
```
* Verify confidence engine behaves predictably under edge cases.
* Verify debug visualizer renders valid annotated video frames.
* Test full pipeline on a sample 5-second test video.
