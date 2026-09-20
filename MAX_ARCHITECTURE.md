# MAX — Target Speaker Isolation & Visual-Audio Association AI

> **Revision note:** This version incorporates four changes to the original
> draft, each called out inline with a `> REVISED:` blockquote where it
> applies:
> 1. Merged the old §8 (lip activity) + §10 (target speaker ID) + §11
>    (AV sync) into one **Audio-Visual Active Speaker Detection (ASD)**
>    component. These three were describing one job — correlating mouth
>    movement with the audio waveform — that a single audio-visual model
>    (TalkNet-ASD / Light-ASD) already does end-to-end, instead of three
>    bespoke pieces of correlation logic you'd have to build and debug
>    separately.
> 2. Added an explicit **voiceprint enrollment & bootstrapping** section
>    (new §10) to handle the case where the target person never speaks
>    alone anywhere in the video — target-speaker separation needs a clean
>    voiceprint to condition on, and the original draft didn't say what
>    happens when one can't be obtained.
> 3. Added a **Phase 0 hardware smoke test** before any real build work,
>    since several model families (speech separation especially) are
>    CUDA-first and MPS support is inconsistent — better to find out a
>    model silently falls back to CPU before you've built three phases on
>    top of it, not after.
> 4. Two smaller additions: an audio/video desync test case in §27, and an
>    explicit note in the confidence engine (§22) that the weighted-sum
>    combiner treats signals as independent when they aren't really —
>    flagged as a known v1 simplification, not a bug to fix immediately.

---

## 1. PROJECT OBJECTIVE

Build a local, modular AI system that can analyze a video containing
multiple people speaking simultaneously and isolate the speech of a
specific person selected by the user.

The system will eventually be integrated into my private AI assistant
**MAX**, which is the voice/vision intelligence inside my Jarvis project.

The core problem:

> When several people are talking at the same time, it is difficult to
> determine who said what and to isolate one person's speech from
> everyone else.

MAX should solve this by combining:

* Computer vision
* Face detection
* Person detection
* Multi-person tracking
* Face tracking
* Clothing/appearance understanding
* Audio-visual active speaker detection (mouth movement ↔ audio correlation)
* Voice activity detection
* Speaker recognition / speaker embeddings
* Audio source separation
* Natural-language target selection
* Confidence scoring
* Temporal tracking

The system must work on an input video and produce a clean representation
of the selected person's speech.

---

## 2. USER EXPERIENCE

The user provides a video containing multiple people.

Example:

There are 5 people:

* Person A: blue T-shirt
* Person B: white shirt
* Person C: black T-shirt and white pants
* Person D: red shirt
* Person E: checked shirt

The user says:

> "Focus on the man wearing the blue T-shirt."

or:

> "I want the voice of the person in black."

or:

> "Focus on that person with the white pants."

or:

> "Get me what the person on the left is saying."

or:

> "Follow that guy."

or:

> "Listen to the woman wearing the red shirt."

MAX must interpret the description and identify the corresponding person
in the video.

---

## 3. COMPLETE PIPELINE

> **REVISED:** collapsed the old separate lip-activity, target-speaker-ID,
> and AV-sync stages into one Active Speaker Detection stage, and added an
> explicit voiceprint enrollment step feeding into separation.

```
INPUT VIDEO
↓
Video decoding
↓
Frame extraction
↓
Person detection
↓
Face detection
↓
Person tracking
↓
Face/person association
↓
Visual attribute extraction
↓
Natural-language target interpretation
↓
Target candidate matching
↓
Target confidence scoring
↓
Audio extraction
↓
Voice Activity Detection
↓
Audio-Visual Active Speaker Detection
   (mouth movement + audio → per-identity speaking-probability timeline;
    replaces separate lip-tracking / sync / speaker-ID stages)
↓
Voiceprint enrollment
   (from high-confidence solo segments; falls back to blind separation +
    ASD as a weak label if no solo segment exists — see §10)
↓
Speech source separation (target speaker extraction, conditioned on
   voiceprint where available)
↓
Target speech enhancement
↓
Speech transcription
↓
Temporal alignment
↓
Confidence estimation
↓
OUTPUT
```

---

## 4. PERSON DETECTION

Detect every visible person in the video.

For each person maintain a persistent track:

```text
person_id
bounding_box
frame_start
frame_end
tracking_confidence
face_available
body_visible
clothing_attributes
position
speaking_probability
speaker_embedding
```

Example:

```json
{
  "person_id": "person_03",
  "appearance": {
    "shirt": "blue",
    "shirt_type": "t-shirt",
    "pants": "black"
  },
  "position": "left-center",
  "tracking_confidence": 0.94
}
```

The IDs must remain stable across frames whenever possible.

If the person temporarily disappears behind another person, attempt track
recovery instead of immediately creating a new identity.

---

## 5. VISUAL DESCRIPTION UNDERSTANDING

The user will NOT necessarily provide exact descriptions.

Support natural descriptions such as:

* "the guy in blue"
* "blue T-shirt man"
* "person wearing white pants"
* "woman on the left"
* "guy standing behind him"
* "person in black"
* "the person closest to the camera"
* "the person sitting down"
* "the person who is talking"
* "that guy"
* "the person with checked clothes"

Convert the user's description into structured attributes.

Example:

User:

> "Focus on the guy wearing a blue T-shirt and black pants."

Convert to:

```json
{
  "shirt_color": "blue",
  "shirt_type": "t-shirt",
  "pants_color": "black",
  "gender_hint": "male"
}
```

Do NOT require every attribute to match.

Use weighted matching.

Example:

```text
shirt color       35%
shirt type        15%
pants color       20%
position           10%
gender hint        5%
face information  15%
```

Weights should be configurable.

---

## 6. TARGET PERSON MATCHING

After extracting the user's description, compare it against every
detected person.

Calculate:

```text
visual_match_score
tracking_score
face_match_score
position_score
overall_target_score
```

Example:

```text
Person 1 → 0.31
Person 2 → 0.92
Person 3 → 0.44
Person 4 → 0.18
```

If one candidate clearly exceeds the others:

```text
Target = Person 2
Confidence = 92%
```

If two people are similarly matched:

```text
Candidate A = 71%
Candidate B = 69%
```

DO NOT silently choose one.

Instead mark the result:

```text
AMBIGUOUS_TARGET
```

The architecture should allow MAX to ask the user for clarification.

---

## 7. FACE + PERSON ASSOCIATION

Associate detected faces with body/person tracks.

Maintain:

```text
person_id
face_id
face_embedding
face_bbox
face_confidence
```

The same person's face and body should remain associated over time.

Handle:

* head rotation
* partial occlusion
* temporary face disappearance
* different camera angles
* movement
* changing lighting

Do not assume that every frame contains a visible face.

---

## 8. AUDIO-VISUAL ACTIVE SPEAKER DETECTION

> **REVISED — this section replaces the original separate "Lip/Mouth
> Activity" and "Audio-Visual Synchronization" sections.** They were
> describing the same underlying signal (does this face's mouth movement
> correlate with the audio right now?), so treat them as one component
> built on a single audio-visual model rather than three hand-tuned
> correlation stages.

For each tracked face, run an audio-visual active speaker detection (ASD)
model — e.g. TalkNet-ASD or Light-ASD — over the face-crop sequence
**together with** the corresponding audio window. The model outputs a
per-frame speaking probability directly, because it's trained to detect
exactly this correlation; you don't need to hand-build a separate
lip-motion tracker and a separate sync-scoring step on top of it.

Output per identity, per time window:

```text
speaking_probability
```

Example:

```text
Person 2:
speaking_probability = 0.91
```

This single signal is what distinguishes:

```text
Person A talking
Person B standing silently
Person C moving their mouth for a non-speech reason (chewing, laughing)
```

from:

```text
Person A actually producing the detected speech.
```

Still treat this as **one signal among several**, not ground truth on its
own — combine it with the audio speaker embedding and tracking stability
in the confidence engine (§22). A model can be confidently wrong when,
e.g., someone is mouthing along to another speaker's words.

If you later want a finer-grained mouth-openness or landmark signal for
the debug visualizer (§15) — separate from the ASD probability — that's
fine to keep as a lightweight secondary output of the same pass. Don't
build it as an independent pipeline stage feeding back into speaker
identification; that's the redundancy this revision removes.

---

## 9. AUDIO PROCESSING

Extract the original audio track from the video.

Perform:

### Stage 1 — Voice Activity Detection

Determine when speech occurs.

```text
00:00.0 - 00:03.2 → silence
00:03.2 - 00:07.8 → speech
00:07.8 - 00:08.4 → silence
```

### Stage 2 — Speaker Representation

Create speaker embeddings for voices detected in the recording.

Represent each speaker as a voice vector.

Do not depend purely on pitch.

Voice identity should consider speaker characteristics represented by the
embedding model.

---

## 10. VOICEPRINT ENROLLMENT & BOOTSTRAPPING

> **NEW SECTION.** The original draft assumed a clean voiceprint would
> always be available to condition target-speaker extraction on. That's
> not guaranteed — someone can be part of an overlapping conversation for
> the entire video and never get a solo moment. This section defines what
> happens then, since it changes the interface of the separator (§12) and
> needs to be decided before you build that phase, not discovered during it.

**Preferred path — clean enrollment:**
Once ASD (§8) reports a window where exactly one identity has high
speaking probability and no other tracked identity does, treat that as a
clean segment. Run a speaker embedding model (ECAPA-TDNN or Resemblyzer)
on it to produce that identity's voiceprint. Accumulate across multiple
clean segments and average/update the embedding as more become available
— don't stop at the first one, since a single short segment can produce a
noisy embedding.

**Fallback path — no clean segment exists:**
If, after scanning the full video, an identity has *no* window where they
speak alone:

1. Run blind source separation (Conv-TasNet or SepFormer) on the
   overlapping segment(s) where ASD indicates they're speaking.
2. Use the ASD-derived timing as a weak label to pick which separated
   stream most plausibly belongs to the target (the stream whose energy
   correlates best with their mouth-movement timing).
3. Extract a *provisional* voiceprint from that stream, explicitly marked
   lower-confidence, and use it to re-run target-speaker extraction as a
   second pass. This can improve output quality even though the initial
   voiceprint was noisy.
4. If separation quality on that provisional stream is itself too low to
   trust (see `LOW_SEPARATION_CONFIDENCE` in §23), surface a new failure
   state rather than silently returning a bad result:

```text
INSUFFICIENT_ENROLLMENT_DATA
```

Document in `MODEL_SELECTION.md` which path was used for a given
identity — this matters for debugging why one person's output is cleaner
than another's in the same video.

---

## 11. OVERLAPPING SPEECH

This system MUST support situations where:

```text
Person A speaks
+
Person B speaks simultaneously
```

The audio separator should attempt to isolate the target speaker,
conditioned on the voiceprint from §10 where one is available.

Use an appropriate neural speech-separation architecture rather than
simply applying frequency filtering.

Frequency-domain processing may be used as part of the pipeline, but DO
NOT assume:

> "Different speaker = different frequency."

Human voices overlap heavily in frequency.

The system should therefore use learned speaker representations and
source separation — and, per §10, have a defined fallback for when a
clean voiceprint isn't available to condition on.

---

## 12. TARGET SPEECH EXTRACTION

Once the target speaker is identified and a voiceprint is available
(clean or provisional, per §10):

```text
Original audio
        ↓
Speaker separation (conditioned on voiceprint where available)
        ↓
Target speaker source
        ↓
Noise reduction
        ↓
Speech enhancement
        ↓
Target speech waveform
```

Preserve the original speech as much as possible.

Do NOT aggressively process the signal to the point where words become
distorted.

The output should include:

```text
target_audio.wav
```

and optionally:

```text
target_audio_enhanced.wav
```

Keep both whenever possible. Also record which enrollment path (§10) was
used, since it affects how much to trust the output.

---

## 13. SPEECH TRANSCRIPTION

After isolating the target speaker, transcribe their speech.

Example:

```text
[00:04.21]
"Hey, are we going to the meeting?"

[00:07.83]
"Yes, I'll be there in ten minutes."
```

Every transcription segment should contain:

```json
{
  "speaker": "person_02",
  "start": 4.21,
  "end": 7.83,
  "text": "Hey, are we going to the meeting?",
  "confidence": 0.94
}
```

---

## 14. TEMPORAL SPEAKER MAP

Create a complete timeline showing who is likely speaking.

Example:

```text
00:00 - 00:03 → nobody
00:03 - 00:06 → Person 1
00:06 - 00:08 → Person 2
00:08 - 00:11 → Person 1 + Person 3
00:11 - 00:15 → Person 2
```

For overlapping speech:

```text
00:08 - 00:11

Person 1 → 0.87
Person 3 → 0.81
```

The system should preserve uncertainty instead of pretending the result
is perfect.

---

## 15. VISUALIZATION / DEBUG MODE

Create a debug mode showing the processing.

Example overlay:

```text
┌──────────────────────────────────────────────┐
│                                              │
│     PERSON 01                 PERSON 02      │
│     [FACE BOX]                [FACE BOX]     │
│        81%                       94%          │
│     NOT SPEAKING               SPEAKING      │
│                                              │
│                         TARGET ★             │
│                         94%                  │
│                                              │
├──────────────────────────────────────────────┤
│ Target speaker: Person 02                    │
│ ASD speaking prob: 93%                       │
│ Speaker embedding match: 88%                 │
│ Enrollment path: clean                       │
│ Target Confidence: 91%                       │
└──────────────────────────────────────────────┘
```

This debug visualization is extremely important during development.

---

## 16. NATURAL-LANGUAGE CONTROL

Expose a simple API/function such as:

```python
focus_on_person(description)
```

Example:

```python
focus_on_person(
    "the man wearing the blue T-shirt and black pants"
)
```

Return:

```json
{
  "target_person": "person_02",
  "confidence": 0.92,
  "status": "confirmed"
}
```

If ambiguous:

```json
{
  "status": "ambiguous",
  "candidates": [
    {
      "person_id": "person_02",
      "confidence": 0.71
    },
    {
      "person_id": "person_04",
      "confidence": 0.68
    }
  ]
}
```

---

## 17. MAX INTEGRATION

The system must be designed as a module that MAX can call.

Example:

```python
result = speaker_focus.process_video(
    video_path,
    user_instruction
)
```

Possible MAX commands:

```text
"Focus on the man in blue."

"Listen to the person with white pants."

"What is that person saying?"

"Follow the guy on the left."

"Give me only that person's voice."

"Transcribe what the woman in red said."
```

MAX should translate the natural-language instruction into the
speaker-focus module.

Do NOT tightly couple this system to the rest of MAX.

It should remain an independent module with a clean API.

---

## 18. PRIVACY / LOCAL PROCESSING

This project is intended to run locally.

Do NOT require cloud APIs.

Do NOT upload videos or audio to external services.

All processing should preferably happen locally on the machine.

Design the architecture so individual models can be replaced with better
local models later.

---

## 19. HARDWARE AWARENESS

The initial development machine is an Apple Silicon Mac.

Optimize for:

* Apple Silicon
* CPU efficiency
* GPU/Metal acceleration where supported
* memory efficiency
* batch processing
* streaming processing where possible

Do not load huge models unnecessarily.

Allow models to be loaded lazily.

For example:

```text
Video received
↓
Load required vision model
↓
Process
↓
Release unnecessary resources
```

> **See Phase 0 in §26** — every model shortlisted here needs to actually
> be confirmed running on this hardware (MPS, or a clean CPU fallback)
> before it's committed to in the architecture doc, not after it's
> already built into a pipeline stage.

---

## 20. MODULAR ARCHITECTURE

> **REVISED:** merged `lip_activity.py` and the old sync logic into
> `active_speaker.py`; added `enrollment.py` for §10.

```text
speaker_focus_ai/
│
├── core/
│   ├── pipeline.py
│   ├── config.py
│   └── types.py
│
├── vision/
│   ├── person_detector.py
│   ├── face_detector.py
│   ├── tracker.py
│   └── attributes.py
│
├── audio/
│   ├── extractor.py
│   ├── vad.py
│   ├── speaker_embedding.py
│   ├── enrollment.py          # NEW — clean + provisional voiceprint bootstrap
│   ├── separator.py
│   └── enhancer.py
│
├── multimodal/
│   ├── target_matcher.py
│   ├── active_speaker.py      # MERGED — replaces lip_activity.py + av_sync.py
│   └── confidence.py
│
├── speech/
│   └── transcription.py
│
├── interface/
│   └── api.py
│
├── debug/
│   └── visualizer.py
│
├── tests/
│
├── requirements.txt
│
└── README.md
```

Keep every component replaceable.

---

## 21. MODEL SELECTION

Do not blindly choose models.

Before implementing each component, evaluate suitable open-source/local
models for:

### Vision

* person detection
* face detection
* tracking
* clothing/attribute recognition

### Audio-visual

* active speaker detection (mouth-motion ↔ audio correlation — this
  single model now covers what was previously split across lip-activity
  and AV-sync)

### Audio

* VAD
* speaker embeddings
* source separation
* speech enhancement

### Speech

* local speech-to-text

For each model, document:

```text
Model
Purpose
Input
Output
Model size
CPU requirements
Apple Silicon (MPS) compatibility — confirmed via Phase 0 smoke test
License
Expected accuracy
Latency
```

Prefer lightweight models initially.

Use heavier models only when they provide a meaningful improvement.

---

## 22. CONFIDENCE ENGINE

Do not create a fake confidence number.

Confidence should be derived from measurable signals.

At minimum track:

```text
visual_match
face_match
speaker_match
asd_speaking_probability
tracking_stability
separation_quality
enrollment_confidence   # clean vs. provisional, per §10
```

Then combine them using a transparent configurable scoring system.

Example:

```python
confidence = weighted_score(
    visual_match,
    speaker_match,
    asd_speaking_probability,
    tracking_stability,
    enrollment_confidence,
)
```

> **REVISED — known simplification, not a bug:** a plain weighted sum
> treats these signals as independent, but they aren't — `speaker_match`
> and `asd_speaking_probability` tend to move together (both degrade
> together when a segment is noisy or overlapping), so a naive sum can
> understate joint uncertainty. This is fine for v1. Once you have
> labeled data from the test set in §27, it's worth revisiting with a
> proper learned combiner (even simple logistic regression over the
> signals) rather than hand-tuned weights. Document this as a known
> limitation in `ARCHITECTURE.md` rather than treating the current scheme
> as final.

Document exactly how the score is calculated.

---

## 23. FAILURE HANDLING

The system must gracefully handle:

### No face visible

```text
FACE_NOT_VISIBLE
```

### Multiple matching people

```text
AMBIGUOUS_TARGET
```

### No speech

```text
NO_SPEECH_DETECTED
```

### Target not speaking

```text
TARGET_NOT_SPEAKING
```

### Severe overlapping speech

```text
LOW_SEPARATION_CONFIDENCE
```

### Poor audio

```text
LOW_AUDIO_QUALITY
```

### Person leaves frame

```text
TARGET_LOST
```

### Insufficient evidence

```text
INSUFFICIENT_CONFIDENCE
```

### No clean voiceprint obtainable

> **NEW — added per §10.**

```text
INSUFFICIENT_ENROLLMENT_DATA
```

Never fabricate a confident result.

---

## 24. IMPORTANT DISTINCTION

The system should distinguish:

```text
PERSON IDENTITY
```

from:

```text
SPEAKER IDENTITY
```

and:

```text
CURRENT SPEAKING STATE
```

These are different concepts.

Example:

Person 2 may be visually identified correctly but currently not
speaking.

Therefore:

```text
person_identity = Person 2
speaking_probability = 0.12
```

Later:

```text
person_identity = Person 2
speaking_probability = 0.94
```

The system must maintain this temporal state.

---

## 25. PROCESSING MODES

Implement three modes.

### MODE 1 — ANALYSIS

Analyze the complete video.

Output:

```text
people
speakers
timeline
target candidates
confidence
transcript
```

### MODE 2 — TARGET EXTRACTION

Find a requested person and generate:

```text
isolated target audio
target transcript
timestamps
confidence
```

### MODE 3 — LIVE / STREAMING

Eventually support:

```text
camera/microphone
↓
real-time people tracking
↓
user selects person
↓
real-time speaker association
↓
real-time target speech
```

Do NOT implement live mode first.

Build the offline video pipeline first and make the architecture capable
of extending to streaming later.

---

## 26. DEVELOPMENT ORDER

> **REVISED:** added Phase 0; merged the old separate lip-activity and
> AV-sync phases into the ASD phase; added an enrollment phase.

Do not attempt to build everything simultaneously.

Build in milestones:

### PHASE 0 — HARDWARE SMOKE TEST (NEW)

Before any pipeline code: for every model shortlisted in
`MODEL_SELECTION.md`, confirm it actually loads and runs on this Apple
Silicon machine — either genuinely accelerated via MPS, or a clean,
acceptable CPU fallback. Several candidate models (speech separation in
particular) are CUDA-first and have real MPS gaps. Finding this out here
costs minutes; finding it out in Phase 10 costs a rebuilt pipeline stage.

### PHASE 1

Video loading + frame extraction

### PHASE 2

Person detection + tracking

### PHASE 3

Face detection + person/face association

### PHASE 4

Visual attribute extraction

### PHASE 5

Natural-language target matching

### PHASE 6

Audio extraction + VAD

### PHASE 7

Speaker embeddings

### PHASE 8

Audio-visual active speaker detection (merged — replaces the old separate
lip-activity and AV-sync phases)

### PHASE 9

Voiceprint enrollment & bootstrapping (new — see §10)

### PHASE 10

Target speaker separation

### PHASE 11

Target transcription

### PHASE 12

Confidence engine

### PHASE 13

Debug visualization

### PHASE 14

MAX integration

### PHASE 15

Optimization

### PHASE 16

Future real-time mode

After each phase, create a test and verify it before moving forward.

---

## 27. TEST DATA

> **REVISED:** added item 13 (AV desync).

Create controlled test videos containing:

1. One person speaking.
2. Two people speaking sequentially.
3. Three people speaking sequentially.
4. Two people talking simultaneously.
5. Three people talking simultaneously.
6. People moving around.
7. Partial face occlusion.
8. Person leaving/re-entering frame.
9. Similar clothing.
10. Similar voices.
11. Background noise.
12. Music/background audio.
13. **Audio/video desync in the source file itself** (common with phone
    recordings and some screen-capture tools) — without a dedicated test
    case for this, a desynced source file will present as a mysterious
    ASD/sync-score bug rather than what it actually is: bad input.

Measure performance separately for each condition.

---

## 28. EVALUATION METRICS

Do not judge the system only by whether the final transcript "looks
correct."

Track:

```text
Person detection accuracy
Tracking stability
Target selection accuracy
Speaker identification accuracy
Active speaker detection accuracy
Speaker separation quality
Word Error Rate
False target selection rate
Enrollment success rate (clean vs. provisional vs. failed)
Latency
Memory usage
CPU usage
```

For source separation, use appropriate audio separation metrics where
possible.

---

## 29. FIRST IMPLEMENTATION REQUIREMENT

Before writing the complete system, create:

```text
ARCHITECTURE.md
MODEL_SELECTION.md
PIPELINE.md
```

Explain:

1. What models are being used.
2. Why each model was selected.
3. Input/output of each component.
4. Dependencies.
5. Expected hardware requirements — confirmed via Phase 0, not assumed.
6. How components communicate.
7. How MAX will integrate with the system.
8. Which components are prototypes and which are production-ready.

Then implement Phase 0, then Phase 1 only.

Do not generate thousands of lines of code immediately.

---

## 30. CORE DESIGN PRINCIPLE

The most important principle is:

> DO NOT TRY TO SOLVE SPEAKER IDENTIFICATION USING ONLY AUDIO OR ONLY
> VISION.

The system should be genuinely multimodal.

Use:

```text
WHO IS THE PERSON?
        +
WHERE ARE THEY?
        +
WHAT DO THEY LOOK LIKE?
        +
DOES THEIR MOUTH MOVEMENT CORRELATE WITH THE AUDIO RIGHT NOW?
        +
IS THEIR VOICE PRESENT?
        +
DOES THEIR VOICE MATCH A RELIABLE ENROLLED VOICEPRINT — OR A PROVISIONAL ONE?
        +
IS THE RESULT STABLE OVER TIME?
```

Only then determine:

```text
TARGET SPEAKER
```

---

## FINAL GOAL

The finished system should allow MAX to receive a command such as:

> "MAX, focus on the guy wearing the blue T-shirt."

Then MAX should:

```text
1. Understand the description.
2. Scan the video.
3. Detect all people.
4. Track them.
5. Identify the blue-shirt person.
6. Calculate candidate confidence.
7. Confirm the target.
8. Track the target through the video.
9. Run audio-visual active speaker detection to determine when the
   target is speaking.
10. Enroll a voiceprint from clean segments, or bootstrap a provisional
    one if none exist.
11. Separate the target's voice from other speakers, conditioned on the
    voiceprint.
12. Enhance the target speech.
13. Transcribe the target speech.
14. Preserve timestamps.
15. Return the isolated audio + transcript + confidence, including
    which enrollment path was used.
```

The system must prioritize **accuracy, uncertainty handling, modularity,
local execution, and real measurable performance** over simply producing
an impressive-looking demo.

Do not claim that a target is correctly identified when the evidence is
insufficient.
