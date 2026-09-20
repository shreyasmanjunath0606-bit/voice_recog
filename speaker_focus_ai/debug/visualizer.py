"""Debug overlay visualizer (§15).

Renders bounding boxes, face boxes, speaking status, target indicator,
and telemetry overlay onto video frames.
"""

from typing import List, Optional, Dict, Any
import numpy as np
import cv2
from speaker_focus_ai.core.types import PersonTrack


class DebugVisualizer:
    """Draws diagnostic overlays on video frames for debugging."""

    def __init__(self, output_video_path: Optional[str] = None, fps: float = 25.0, frame_size: tuple = (1920, 1080)):
        self.output_video_path = output_video_path
        self.fps = fps
        self.frame_size = frame_size
        self.writer = None
        if self.output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.output_video_path, fourcc, self.fps, self.frame_size)

    def draw_frame_overlay(
        self,
        frame: np.ndarray,
        tracks: List[PersonTrack],
        target_person_id: Optional[str],
        telemetry: Dict[str, Any]
    ) -> np.ndarray:
        """Annotate frame with tracking boxes, speaking probability, target marker, and HUD."""
        out_frame = frame.copy()
        
        # Colors (BGR)
        COLOR_TARGET = (0, 255, 0) # Green for target
        COLOR_OTHER = (255, 0, 0) # Blue for others
        COLOR_TEXT = (255, 255, 255)
        
        for track in tracks:
            is_target = track.person_id == target_person_id
            color = COLOR_TARGET if is_target else COLOR_OTHER
            
            # Draw person bbox
            pb = track.bounding_box
            cv2.rectangle(out_frame, (int(pb.x1), int(pb.y1)), (int(pb.x2), int(pb.y2)), color, 2)
            
            # Label
            label = f"ID: {track.person_id}"
            if is_target:
                label += " (TARGET ★)"
            
            cv2.putText(out_frame, label, (int(pb.x1), max(0, int(pb.y1) - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw face bbox and speaking status if available
            if track.associated_face:
                fb = track.associated_face.bbox
                cv2.rectangle(out_frame, (int(fb.x1), int(fb.y1)), (int(fb.x2), int(fb.y2)), (0, 255, 255), 1)
                speak_prob = track.speaking_probability
                speak_text = f"SPEAKING {int(speak_prob*100)}%" if speak_prob > 0.5 else f"NOT SPEAKING {int(speak_prob*100)}%"
                cv2.putText(out_frame, speak_text, (int(fb.x1), max(0, int(fb.y1) - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Draw Telemetry HUD
        hud_bg_color = (0, 0, 0)
        alpha = 0.6
        overlay = out_frame.copy()
        cv2.rectangle(overlay, (0, out_frame.shape[0] - 100), (out_frame.shape[1], out_frame.shape[0]), hud_bg_color, -1)
        out_frame = cv2.addWeighted(overlay, alpha, out_frame, 1 - alpha, 0)
        
        hud_y = out_frame.shape[0] - 70
        cv2.putText(out_frame, f"Overall Confidence: {telemetry.get('overall_confidence', 0.0):.2f}", 
                    (20, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
        cv2.putText(out_frame, f"Enrollment: {telemetry.get('enrollment_path', 'None')}", 
                    (20, hud_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
        cv2.putText(out_frame, f"ASD Prob: {telemetry.get('asd_prob', 0.0):.2f}", 
                    (400, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
        
        if self.writer is not None:
            # Resize if necessary to match writer
            if (out_frame.shape[1], out_frame.shape[0]) != self.frame_size:
                out_frame_write = cv2.resize(out_frame, self.frame_size)
            else:
                out_frame_write = out_frame
            self.writer.write(out_frame_write)
            
        return out_frame
        
    def release(self):
        if self.writer is not None:
            self.writer.release()
