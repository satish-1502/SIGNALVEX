"""
tracker.py — Multi-Object Tracking Module for SIGNALVEX
=======================================================
Tracks detected persons across successive frames using ByteTrack.
Assigns consistent Track IDs to reduce duplicate person counts.
Includes a fallback tracker if ByteTrack or lap solver encounters issues.
"""

from typing import List, Dict, Any, Optional
import numpy as np


class HumanTracker:
    """
    Tracks detected humans across video frames using Ultralytics ByteTrack integration
    or an internal fallback tracker.
    """

    def __init__(self, model_instance=None, tracker_type: str = "bytetrack.yaml", conf_threshold: float = 0.35):
        """
        Args:
            model_instance: Ultralytics YOLO model instance from HumanDetector
            tracker_type: Tracker configuration file, default 'bytetrack.yaml'
            conf_threshold: Minimum confidence threshold
        """
        self.model = model_instance
        self.tracker_type = tracker_type
        self.conf_threshold = conf_threshold
        self.fallback_next_id = 1
        self.active_tracks = {}  # {track_id: bbox} for fallback tracker

    def track(self, frame: np.ndarray, conf_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Run tracking on the given video frame.
        Maintains persistent track IDs across successive frames.

        Returns list of track dictionaries:
        [
            {
                "track_id": 1,
                "bbox": [x1, y1, x2, y2],
                "confidence": 0.82,
                "class_id": 0,
                "class_name": "person"
            }, ...
        ]
        """
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold

        # Primary route: Ultralytics YOLO tracker
        if self.model is not None:
            try:
                results = self.model.track(
                    frame,
                    persist=True,
                    classes=[0],
                    conf=conf,
                    tracker=self.tracker_type,
                    verbose=False
                )

                tracks = []
                if results and len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    for box in boxes:
                        score = float(box.conf[0].item())
                        cls_id = int(box.cls[0].item())
                        if cls_id != 0 or score < conf:
                            continue

                        # Extract track ID if assigned
                        if box.id is not None:
                            track_id = int(box.id[0].item())
                        else:
                            # If tracker hasn't assigned an ID yet, use fallback temporary ID
                            track_id = self.fallback_next_id
                            self.fallback_next_id += 1

                        coords = [int(round(c.item())) for c in box.xyxy[0]]
                        tracks.append({
                            "track_id": track_id,
                            "bbox": coords,
                            "confidence": score,
                            "class_id": 0,
                            "class_name": "person"
                        })
                return tracks
            except Exception as e:
                # If Ultralytics track fails (e.g. tracker yaml or lap missing), fall back to detection + centroid match
                print(f"[SIGNALVEX Tracker] Ultralytics tracking error: {e}. Falling back to internal tracker.")

        # Fallback tracker: run simple detector and match by IoU/proximity
        return self._fallback_track(frame, conf)

    def _fallback_track(self, frame: np.ndarray, conf: float) -> List[Dict[str, Any]]:
        """Fallback tracking using IoU-based bounding box association."""
        if self.model is None:
            return []

        res = self.model(frame, classes=[0], conf=conf, verbose=False)
        current_dets = []
        if res and len(res) > 0 and res[0].boxes is not None:
            for box in res[0].boxes:
                score = float(box.conf[0].item())
                if score >= conf:
                    coords = [int(round(c.item())) for c in box.xyxy[0]]
                    current_dets.append((coords, score))

        matched_tracks = []
        new_active = {}

        for coords, score in current_dets:
            matched_id = None
            best_iou = 0.25  # minimum overlap threshold

            for tid, prev_box in self.active_tracks.items():
                iou = self._calc_iou(coords, prev_box)
                if iou > best_iou:
                    best_iou = iou
                    matched_id = tid

            if matched_id is None:
                matched_id = self.fallback_next_id
                self.fallback_next_id += 1

            new_active[matched_id] = coords
            matched_tracks.append({
                "track_id": matched_id,
                "bbox": coords,
                "confidence": score,
                "class_id": 0,
                "class_name": "person"
            })

        self.active_tracks = new_active
        return matched_tracks

    @staticmethod
    def _calc_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter_area = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        union_area = boxAArea + boxBArea - inter_area
        return inter_area / float(union_area) if union_area > 0 else 0.0

    def reset(self):
        """Reset tracker state for new video."""
        self.fallback_next_id = 1
        self.active_tracks.clear()
        if self.model is not None and hasattr(self.model, "predictor") and self.model.predictor is not None:
            if hasattr(self.model.predictor, "trackers"):
                self.model.predictor.trackers = None
