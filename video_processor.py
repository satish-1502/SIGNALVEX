"""
video_processor.py — Video Pipeline & Annotation Module for SIGNALVEX
====================================================================
Handles:
1. Video file validation, OpenCV capture, and metadata extraction.
2. Frame preprocessing and resizing for smooth processing.
3. Overlay annotation: Bounding boxes, confidence %, track IDs, frame timestamps.
4. Output video encoding via cv2.VideoWriter (MP4).
5. Progress callback integration for Streamlit UI.
"""

import time
import os
from typing import Dict, Any, Callable, Optional, Tuple
import cv2
import numpy as np

from tracker import HumanTracker
from logger import DetectionLogger


class VideoProcessor:
    """
    Orchestrates the frame-by-frame processing pipeline.
    """

    def __init__(self, tracker: HumanTracker, logger: DetectionLogger):
        self.tracker = tracker
        self.logger = logger

    @staticmethod
    def get_video_info(video_path: str) -> Dict[str, Any]:
        """
        Validates video file and returns duration, resolution, total frames, and FPS.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file '{os.path.basename(video_path)}'. Format may be corrupted or unsupported.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0.0

        cap.release()

        return {
            "file_name": os.path.basename(video_path),
            "width": width,
            "height": height,
            "fps": round(fps, 2),
            "total_frames": total_frames,
            "duration_sec": round(duration_sec, 2),
            "duration_formatted": f"{int(duration_sec // 60):02d}:{int(duration_sec % 60):02d}"
        }

    @staticmethod
    def annotate_frame(
        frame: np.ndarray,
        tracks: list,
        frame_idx: int,
        timestamp_sec: float,
        low_conf_threshold: float = 0.50
    ) -> np.ndarray:
        """
        Draws bounding boxes, tracking labels, confidence, and timestamp banner onto frame.
        Uses Cyan/Green for confident detections, and Amber/Orange for uncertain ones (per presentation).
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Top banner with SIGNALVEX telemetry status
        banner_h = 36
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 25, 35), -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

        banner_text = f"SIGNALVEX RGB | Frame: {frame_idx:04d} | Time: {timestamp_sec:.2f}s | Active Tracks: {len(tracks)}"
        cv2.putText(annotated, banner_text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (230, 240, 250), 2, cv2.LINE_AA)

        # Draw each tracked human
        for trk in tracks:
            bbox = trk["bbox"]
            track_id = trk["track_id"]
            conf = trk["confidence"]
            x1, y1, x2, y2 = bbox

            # Presentation color rule: Low confidence alert (< 0.50) in amber/orange, normal in cyan/lime
            if conf < low_conf_threshold:
                box_color = (0, 165, 255)  # Amber / Orange (BGR)
                tag_prefix = "REV"
            else:
                box_color = (0, 230, 115)  # Emerald / Lime Green (BGR)
                tag_prefix = "ID"

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

            # Label text
            label = f"{tag_prefix} #{track_id} | {conf*100:.1f}%"
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)

            # Draw filled tag background above box
            tag_y1 = max(banner_h + 5, y1 - text_h - 8)
            tag_y2 = tag_y1 + text_h + 6
            tag_x2 = min(w - 2, x1 + text_w + 10)
            cv2.rectangle(annotated, (x1, tag_y1), (tag_x2, tag_y2), box_color, -1)

            # Dark text inside bright tag
            cv2.putText(
                annotated,
                label,
                (x1 + 4, tag_y2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (10, 10, 15),
                1,
                cv2.LINE_AA
            )

        return annotated

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        conf_threshold: float = 0.35,
        max_frames: Optional[int] = None,
        progress_callback: Optional[Callable[[float, np.ndarray, Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end processing:
        - Reads frames
        - Tracks persons using YOLO + ByteTrack
        - Logs detections
        - Annotates frames
        - Saves output video file if output_path is provided
        - Notifies progress_callback
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames and max_frames > 0:
            total_frames = min(total_frames, max_frames)

        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup video writer if output path is requested
        writer = None
        if output_path:
            out_dir = os.path.dirname(output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Reset tracker and logger state for clean run
        self.tracker.reset()
        self.logger.clear()

        start_time = time.time()
        frame_idx = 0
        processed_count = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if max_frames and frame_idx > max_frames:
                    break

                timestamp_sec = (frame_idx - 1) / fps

                # Run tracking (filters for class 0 person)
                tracks = self.tracker.track(frame, conf_threshold=conf_threshold)

                # Log each tracked detection
                for trk in tracks:
                    self.logger.log_detection(
                        frame_idx=frame_idx,
                        timestamp_sec=timestamp_sec,
                        track_id=trk["track_id"],
                        confidence=trk["confidence"],
                        bbox=trk["bbox"]
                    )

                # Annotate frame
                annotated_frame = self.annotate_frame(
                    frame,
                    tracks,
                    frame_idx=frame_idx,
                    timestamp_sec=timestamp_sec
                )

                if writer is not None:
                    writer.write(annotated_frame)

                processed_count += 1

                # Update callback
                if progress_callback:
                    progress_ratio = processed_count / total_frames if total_frames > 0 else 1.0
                    stats_so_far = {
                        "frame_idx": frame_idx,
                        "processed_count": processed_count,
                        "total_frames": total_frames,
                        "current_tracks": len(tracks),
                        "total_detections": len(self.logger.records)
                    }
                    progress_callback(min(1.0, progress_ratio), annotated_frame, stats_so_far)

        finally:
            cap.release()
            if writer is not None:
                writer.release()

        elapsed = time.time() - start_time
        measured_fps = round(processed_count / elapsed, 2) if elapsed > 0 else 0.0

        summary = self.logger.get_summary_stats()
        summary.update({
            "frames_processed": processed_count,
            "processing_time_sec": round(elapsed, 2),
            "measured_fps": measured_fps,
            "output_video_path": output_path if (output_path and os.path.exists(output_path)) else None
        })

        return summary
