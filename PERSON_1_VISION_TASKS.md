# 🧑‍💻 PERSON 1: Computer Vision & Target Identification Lead

**Title:** Computer Vision & Visual Description Lead ("The Eyes")  
**Git Branch:** `feat/vision-tracking`  
**Base Folder:** `speaker_focus_ai/vision/` & `speaker_focus_ai/multimodal/target_matcher.py`

---

## 🎯 Primary Mission
Detect every person in the video, maintain persistent tracks across occlusions, associate detected faces with bodies, extract visual clothing/position attributes, and interpret natural language descriptions (e.g., *"the man in the blue shirt"*) to identify the target candidate without silent guessing.

---

## 📂 Owned Files & Responsibilities

| File Path | Description | Spec Section |
| :--- | :--- | :--- |
| `speaker_focus_ai/vision/person_detector.py` | Detect human bodies in frames (YOLOv8n / RT-DETR) | §4 |
| `speaker_focus_ai/vision/face_detector.py` | Detect faces and associate with body boxes (MediaPipe Face) | §7 |
| `speaker_focus_ai/vision/tracker.py` | Maintain stable `person_id` across frames (ByteTrack / BoT-SORT) | §4 |
| `speaker_focus_ai/vision/attributes.py` | Extract clothing colors, garment types, spatial positions | §4, §5 |
| `speaker_focus_ai/multimodal/target_matcher.py` | Parse user query & score candidates (handles `AMBIGUOUS_TARGET`) | §5, §6 |
| `speaker_focus_ai/tests/test_vision.py` | Unit and integration tests for vision modules | §26 |

---

## 📋 Step-by-Step Task Checklist

### Milestone 1: Phase 1 — Video Loading & Frame Extraction
- [ ] Implement video frame generator in `speaker_focus_ai/vision/person_detector.py` (or utility) using OpenCV `cv2.VideoCapture` or PyAV.
- [ ] Ensure extraction preserves native FPS, frame timestamps (in seconds), and frame dimensions `(H, W, 3)`.

### Milestone 2: Phase 2 — Person Detection & Tracking
- [ ] Integrate **YOLOv8n** (`ultralytics`) configured for `classes=[0]` (person only).
- [ ] Hook up **ByteTrack** in `tracker.py` to assign persistent `person_id` tracks (e.g. `person_01`, `person_02`).
- [ ] Handle track recovery: if a person is occluded for < 1.5 seconds, keep the same identity rather than spawning a new ID.
- [ ] Return standard `List[PersonTrack]` dataclass objects defined in `speaker_focus_ai/core/types.py`.

### Milestone 3: Phase 3 — Face Detection & Body Association
- [ ] Integrate **MediaPipe Face Detection** (or RetinaFace) in `face_detector.py`.
- [ ] Implement anatomical association rule: match face bounding box inside the upper 50% of the person's bounding box.
- [ ] Populate `PersonTrack.face_available` and `PersonTrack.associated_face`.
- [ ] Handle cases where face turns away or is occluded (`face_available = False`).

### Milestone 4: Phase 4 — Visual Attribute Extraction
- [ ] In `attributes.py`, extract upper body crop (shirt) and lower body crop (pants).
- [ ] Implement color classification (using HSV color binning, K-Means clustering, or lightweight CLIP text-image similarity).
- [ ] Compute spatial tags: `left`, `center`, `right`, `closest to camera` (based on bounding box area).
- [ ] Output structured `ClothingAttributes(shirt_color=..., shirt_type=..., pants_color=...)`.

### Milestone 5: Phase 5 — Natural Language Target Matching
- [ ] In `target_matcher.py`, parse natural queries:
  - *"focus on the guy wearing a blue t-shirt"*
  - *"person on the left"*
  - *"the woman in red"*
- [ ] Calculate weighted candidate score:
  - Shirt color (35%)
  - Shirt type (15%)
  - Pants color (20%)
  - Spatial position (10%)
  - Face available (15%)
  - Gender hint (5%)
- [ ] **Crucial Rule (§6):** If the score difference between top 2 candidates $\le 0.08$, set `status = TargetStatus.AMBIGUOUS` and failure reason `AMBIGUOUS_TARGET`. Do not silently guess!

---

## 🤝 Cross-Team Integration Handoffs

1. **Handoff to Person 3 (Active Speaker Detection):**
   * Provide the crop sequence of each tracked person's face `(T, H, W, 3)` alongside frame timestamps. Person 3 needs these face crops to run TalkNet-ASD.
2. **Handoff to Person 3 (Pipeline Orchestrator):**
   * Provide the confirmed `target_person_id` and candidate list so the pipeline knows whose speech to isolate.

---

## 🧪 Verification & Testing
Run your tests locally:
```bash
python3 -m unittest speaker_focus_ai/tests/test_vision.py
```
* Test with a short 5-second multi-person video clip.
* Verify `person_id` stays constant when two people walk past each other.
