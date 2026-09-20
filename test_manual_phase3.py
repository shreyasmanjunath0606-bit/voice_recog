"""Method 2: Manual real-video visual test for Phase 3 (Face Detection).

Runs Phase 2 (Tracking) and Phase 3 (Face Association) on a real video
and saves the annotated output to 'test_output_phase3.mp4'.
"""

import cv2
import os
import urllib.request

from speaker_focus_ai.vision.person_detector import read_video_frames, get_video_metadata
from speaker_focus_ai.vision.tracker import MultiPersonTracker
from speaker_focus_ai.vision.face_detector import FaceDetector

VIDEO_PATH = "test_real.mp4"
OUTPUT_PATH = "test_output_phase3.mp4"

print("=" * 60)
print(" Phase 3 Visual Test: Face Detection & Body Association")
print("=" * 60)

# Ensure we have the test video from Phase 2
if not os.path.exists(VIDEO_PATH):
    print("Downloading sample video...")
    urllib.request.urlretrieve(
        "https://filesamples.com/samples/video/mp4/sample_640x360.mp4", 
        VIDEO_PATH
    )

meta = get_video_metadata(VIDEO_PATH)
print(f"\n[1/3] Video Info: {meta.fps:.1f} FPS | {meta.duration_sec:.1f}s | {meta.width}x{meta.height}")

tracker = MultiPersonTracker()
face_detector = FaceDetector()

print("\n[2/3] Processing video frames with YuNet & ByteTrack...")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, meta.fps, (meta.width, meta.height))

faces_found = 0

for frame_idx, timestamp_sec, frame in read_video_frames(VIDEO_PATH, max_frames=200): # first ~6 seconds
    # Phase 2: Detect & Track Bodies
    tracks = tracker.track_frame(frame, frame_idx, timestamp_sec)
    
    # Phase 3: Detect Faces & Associate
    faces = face_detector.detect_faces(frame)
    tracks = face_detector.associate_faces_with_persons(faces, tracks)
    
    # Draw results
    for p in tracks:
        # Draw Body Box (Green)
        bx1, by1 = int(p.bounding_box.x1), int(p.bounding_box.y1)
        bx2, by2 = int(p.bounding_box.x2), int(p.bounding_box.y2)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 100), 2)
        
        # Label with Person ID and Position
        label = f"{p.person_id} ({p.position})"
        cv2.putText(frame, label, (bx1, max(by1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 2)

        # Draw Face Box (Magenta) if available
        if p.face_available and p.associated_face:
            faces_found += 1
            fx1, fy1 = int(p.associated_face.bbox.x1), int(p.associated_face.bbox.y1)
            fx2, fy2 = int(p.associated_face.bbox.x2), int(p.associated_face.bbox.y2)
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 0, 255), 2)
            
            # Extract and show the face crop in the corner as a demo!
            crop = face_detector.extract_face_crop(frame, p.associated_face, margin=0.2)
            if crop is not None and crop.size > 0:
                crop_h, crop_w = crop.shape[:2]
                # Put it in the top right corner
                frame[10:10+crop_h, meta.width-crop_w-10:meta.width-10] = crop
                cv2.rectangle(frame, (meta.width-crop_w-10, 10), (meta.width-10, 10+crop_h), (255, 0, 255), 2)
                cv2.putText(frame, "Face Crop", (meta.width-crop_w-10, 30+crop_h), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

    out_writer.write(frame)

    if frame_idx % 30 == 0:
        print(f"    [{timestamp_sec:.2f}s] Faces associated: {sum(1 for p in tracks if p.face_available)}")

out_writer.release()
print(f"\n[3/3] DONE! Total face associations made: {faces_found}")
print(f"\n✅ Annotated video saved to '{OUTPUT_PATH}'")
print("Open this file to see both Body Tracking and Face Tracking working together!")
