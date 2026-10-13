"""
logger.py — Detection Logging and CSV Export Module for SIGNALVEX
================================================================
Maintains structured detection records across video frames.
Tracks review status (Keep, Reject, Needs Review) per detection or track.
Exports data cleanly to CSV format according to search-and-rescue specifications.
"""

from typing import List, Dict, Any, Optional
import pandas as pd


class DetectionLogger:
    """
    Manages detection records, review decisions, and CSV exports.
    """

    # Valid operator review states
    STATUS_NEEDS_REVIEW = "Needs Review"
    STATUS_KEEP = "Keep"
    STATUS_REJECT = "Reject"

    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def clear(self):
        """Clears all logged records."""
        self.records.clear()

    def log_detection(
        self,
        frame_idx: int,
        timestamp_sec: float,
        track_id: int,
        confidence: float,
        bbox: List[int],
        initial_status: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Log an individual detected person record.
        Auto-assigns review status based on confidence if not specified.
        (e.g., lower confidence gets 'Needs Review' by default).
        """
        if initial_status is None:
            # Low confidence alert heuristic from presentation (e.g. < 0.50 flagged as Needs Review)
            if confidence < 0.50:
                initial_status = self.STATUS_NEEDS_REVIEW
            else:
                initial_status = self.STATUS_KEEP

        record_id = f"det_{frame_idx}_{track_id}_{len(self.records)}"
        entry = {
            "record_id": record_id,
            "frame_number": int(frame_idx),
            "timestamp_seconds": round(float(timestamp_sec), 3),
            "track_id": int(track_id),
            "confidence": round(float(confidence), 4),
            "confidence_percent": round(float(confidence) * 100, 1),
            "x1": int(bbox[0]),
            "y1": int(bbox[1]),
            "x2": int(bbox[2]),
            "y2": int(bbox[3]),
            "bbox_str": f"[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]",
            "review_status": initial_status,
        }
        self.records.append(entry)
        return entry

    def update_status(self, record_id: str, new_status: str) -> bool:
        """Update review decision for an individual record."""
        for rec in self.records:
            if rec["record_id"] == record_id:
                rec["review_status"] = new_status
                return True
        return False

    def update_status_by_track(self, track_id: int, new_status: str):
        """Update review decision for all records of a particular track ID."""
        for rec in self.records:
            if rec["track_id"] == track_id:
                rec["review_status"] = new_status

    def to_dataframe(self) -> pd.DataFrame:
        """Convert recorded detections to a pandas DataFrame."""
        if not self.records:
            return pd.DataFrame(columns=[
                "record_id", "frame_number", "timestamp_seconds", "track_id",
                "confidence", "confidence_percent", "x1", "y1", "x2", "y2",
                "bbox_str", "review_status"
            ])
        return pd.DataFrame(self.records)

    def to_csv(self, file_path: Optional[str] = None) -> str:
        """Export detection log to CSV string or file."""
        df = self.to_dataframe()
        # Select and format user-facing columns for search-and-rescue reporting
        export_cols = [
            "frame_number", "timestamp_seconds", "track_id",
            "confidence", "x1", "y1", "x2", "y2", "review_status"
        ]
        export_df = df[export_cols] if not df.empty else pd.DataFrame(columns=export_cols)

        if file_path:
            export_df.to_csv(file_path, index=False)
        return export_df.to_csv(index=False)

    def get_summary_stats(self) -> Dict[str, Any]:
        """Compute aggregate summary counts."""
        df = self.to_dataframe()
        if df.empty:
            return {
                "total_detections": 0,
                "approx_distinct_tracks": 0,
                "needs_review_count": 0,
                "keep_count": 0,
                "reject_count": 0,
                "avg_confidence": 0.0
            }

        distinct_tracks = df["track_id"].nunique()
        needs_review = len(df[df["review_status"] == self.STATUS_NEEDS_REVIEW])
        keep = len(df[df["review_status"] == self.STATUS_KEEP])
        reject = len(df[df["review_status"] == self.STATUS_REJECT])
        avg_conf = df["confidence"].mean()

        return {
            "total_detections": len(df),
            "approx_distinct_tracks": int(distinct_tracks),
            "needs_review_count": int(needs_review),
            "keep_count": int(keep),
            "reject_count": int(reject),
            "avg_confidence": round(float(avg_conf), 3)
        }
