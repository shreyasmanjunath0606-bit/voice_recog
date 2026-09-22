"""End-to-End Multimodal Pipeline Demo."""

import json
from speaker_focus_ai.interface.api import process_video

import sys

if len(sys.argv) < 3:
    print("Usage: python run_demo.py <path_to_video.mp4> \"<instruction text>\"")
    print("Example: python run_demo.py test_real.mp4 \"the guy in the blue shirt on the left\"")
    sys.exit(1)

VIDEO_PATH = sys.argv[1]
INSTRUCTION = sys.argv[2]

print("=====================================================")
print("🤖 MAX: Speaker Focus AI - Full Pipeline Demo")
print("=====================================================")
print(f"Video: {VIDEO_PATH}")
print(f"User Instruction: \"{INSTRUCTION}\"")
print("Initializing all Vision, Audio, and Multimodal models...")
print("(This may take a minute on the first run as models load)")
print("=====================================================\n")

# Run the full end-to-end pipeline
result = process_video(
    video_path=VIDEO_PATH,
    user_instruction=INSTRUCTION,
    output_dir="./demo_output"
)

print("\n=====================================================")
print("✅ Pipeline Completed!")
print("=====================================================")
print("Final Output Payload:")
print(json.dumps(result, indent=2))
print("=====================================================")
print(f"Check the './demo_output' folder for:")
print("1. Target isolated audio (.wav)")
print("2. Enhanced isolated audio (.wav)")
print("3. Telemetry Visualizer Video (.mp4)")
