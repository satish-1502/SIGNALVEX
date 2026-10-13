"""
detector.py — YOLO Human Detection Module for SIGNALVEX
======================================================
Loads a pretrained YOLO model (e.g., YOLOv8n) and performs inference on frames.
Strictly filters detections for the 'person' class (COCO class 0).
Supports fallback download and local model caching.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np


class HumanDetector:
    """
    YOLO-based person detector.
    Pretrained on COCO dataset where class index 0 represents 'person'.
    """

    PERSON_CLASS_ID = 0

    def __init__(self, model_path: str = "models/yolov8n.pt", conf_threshold: float = 0.35):
        """
        Initialize detector with model path and default confidence threshold.
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None
        self._load_model()

    def _load_model(self):
        """Loads Ultralytics YOLO model with fallback handling."""
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "Ultralytics package is not installed. Please run 'pip install ultralytics'."
            ) from e

        # Ensure model directory exists
        model_file = Path(self.model_path)
        model_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            # If path exists or model name is recognizable, YOLO will load or download automatically
            self.model = YOLO(self.model_path)
        except Exception as e:
            # Fallback: try default yolov8n if custom path had an issue
            fallback_name = "yolov8n.pt"
            print(f"[SIGNALVEX Detector] Failed loading '{self.model_path}': {e}. Attempting fallback '{fallback_name}'...")
            try:
                self.model = YOLO(fallback_name)
                self.model_path = fallback_name
            except Exception as fallback_err:
                raise RuntimeError(
                    f"Failed to load YOLO model: {fallback_err}. "
                    "Ensure internet connection is available for the initial model download or place 'yolov8n.pt' in the models/ directory."
                ) from fallback_err

    def detect(self, frame: np.ndarray, conf_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Run inference on a single BGR image/frame.
        Filters strictly for class 0 (person) and confidence >= conf_threshold.

        Returns a list of detection dictionaries:
        [
            {
                "bbox": [x1, y1, x2, y2],
                "confidence": 0.85,
                "class_id": 0,
                "class_name": "person"
            }, ...
        ]
        """
        if self.model is None:
            raise RuntimeError("YOLO model is not initialized.")

        conf = conf_threshold if conf_threshold is not None else self.conf_threshold

        # Run inference filtering for person class (class 0)
        results = self.model(
            frame,
            classes=[self.PERSON_CLASS_ID],
            conf=conf,
            verbose=False
        )

        detections = []
        if not results:
            return detections

        first_res = results[0]
        boxes = first_res.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            cls_id = int(box.cls[0].item())
            score = float(box.conf[0].item())
            if cls_id == self.PERSON_CLASS_ID and score >= conf:
                coords = [int(round(c.item())) for c in box.xyxy[0]]
                detections.append({
                    "bbox": coords,
                    "confidence": score,
                    "class_id": cls_id,
                    "class_name": "person"
                })

        return detections
