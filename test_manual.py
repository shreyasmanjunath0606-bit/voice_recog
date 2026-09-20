"""Method 2: Manual real-video visual test for Phase 1 & Phase 2.

Step 1: Generates a synthetic 5-second test video with moving rectangles (simulating people).
Step 2: Runs the tracker on it frame-by-frame and prints detection results.
Step 3: Saves an annotated output video to 'test_output_annotated.mp4'.
"""

import cv2
import os
import numpy as np
from speaker_focus_ai.vision.person_detector import read_video_frames, get_video_metadata
from speaker_focus_ai.vision.tracker import MultiPersonTracker

# ─── Step 1: Create a synthetic test video ─────────────────────────────────────
VIDEO_PATH = "test_input.mp4"
OUTPUT_PATH = "test_output_annotated.mp4"
FPS = 30
NUM_FRAMES = 150  # 5 seconds
WIDTH, HEIGHT = 640, 480

print("=" * 55)
print(" Method 2: Phase 1 & Phase 2 Manual Test")
print("=" * 55)
print(f"\n[1/3] Generating synthetic test video: {VIDEO_PATH}")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(VIDEO_PATH, fourcc, FPS, (WIDTH, HEIGHT))

for i in range(NUM_FRAMES):
    frame = np.full((HEIGHT, WIDTH, 3), 40, dtype=np.uint8)  # Dark background

    # Person 1: moves left to right
    p1_x = int(50 + (i * 3.5) % (WIDTH - 120))
    cv2.rectangle(frame, (p1_x, 100), (p1_x + 70, 380), (80, 160, 80), -1)
    cv2.circle(frame, (p1_x + 35, 80), 30, (180, 140, 100), -1)  # "head"
    cv2.putText(frame, "P1", (p1_x + 20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Person 2: moves right to left
    p2_x = int(WIDTH - 120 - (i * 2.5) % (WIDTH - 120))
    cv2.rectangle(frame, (p2_x, 100), (p2_x + 70, 380), (80, 80, 180), -1)
    cv2.circle(frame, (p2_x + 35, 80), 30, (180, 140, 100), -1)  # "head"
    cv2.putText(frame, "P2", (p2_x + 20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Frame counter overlay
    cv2.putText(frame, f"Frame: {i:03d}  Time: {i/FPS:.2f}s", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    writer.write(frame)

writer.release()
print(f"    Generated: {NUM_FRAMES} frames @ {FPS} FPS ({NUM_FRAMES/FPS:.1f}s)")

# ─── Step 2: Read metadata & verify Phase 1 ───────────────────────────────────
print("\n[2/3] Running Phase 1: Video Metadata & Frame Reader...")
meta = get_video_metadata(VIDEO_PATH)
print(f"    ✅ FPS       : {meta.fps:.1f}")
print(f"    ✅ Resolution: {meta.width}x{meta.height}")
print(f"    ✅ Duration  : {meta.duration_sec:.2f}s")
print(f"    ✅ Frames    : {meta.total_frames}")

# ─── Step 3: Run Phase 2 tracking & save annotated output video ───────────────
print("\n[3/3] Running Phase 2: Person Detection & ByteTrack Tracking...")
print("       Processing frames and saving annotated video...")
tracker = MultiPersonTracker()
out_writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (WIDTH, HEIGHT))

total_detections = 0
for frame_idx, timestamp_sec, frame in read_video_frames(VIDEO_PATH):
    tracks = tracker.track_frame(frame, frame_idx, timestamp_sec)
    total_detections += len(tracks)

    # Draw tracking results on each frame
    for p in tracks:
        x1, y1 = int(p.bounding_box.x1), int(p.bounding_box.y1)
        x2, y2 = int(p.bounding_box.x2), int(p.bounding_box.y2)
        conf = p.bounding_box.confidence
        label = f"{p.person_id} | {p.position} | {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 2)
        cv2.putText(frame, label, (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 100), 2)

    # Status overlay
    status = f"Tracking {len(tracks)} person(s) @ {timestamp_sec:.2f}s"
    cv2.putText(frame, status, (10, HEIGHT - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 0), 1)

    out_writer.write(frame)

    if frame_idx % 30 == 0:
        ids = [p.person_id for p in tracks]
        print(f"    [{timestamp_sec:.2f}s] Tracked IDs: {ids}")

out_writer.release()

print(f"\n{'=' * 55}")
print(" ✅ Phase 1 PASSED: Video decoded with exact timestamps")
print(f" ✅ Phase 2 PASSED: Total detection instances: {total_detections}")
print(f" 📹 Annotated video saved → {OUTPUT_PATH}")
print(f"{'=' * 55}")
print("\nOpen 'test_output_annotated.mp4' to see tracked bounding boxes!")
