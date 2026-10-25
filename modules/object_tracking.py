"""
VisionGuard — Object Tracking Module
Centroid and IoU tracker maintaining vehicle identities across frames,
computing motion vectors (for direction and stop-line crossing), and suppressing duplicates.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional


class TrackedObject:
    """Represents an active tracked entity across consecutive video frames."""

    def __init__(self, track_id: str, class_name: str, bbox: List[int], conf: float):
        self.track_id = track_id
        self.class_name = class_name
        self.bbox = [int(v) for v in bbox]
        self.conf = conf
        self.centroid = self._calc_centroid(self.bbox)
        self.history = [self.centroid]
        self.disappeared = 0
        self.frames_active = 1
        self.violations_flagged = set()

        # Motion & Direction properties
        self.direction = "stationary"  # 'downward', 'upward', 'left', 'right', 'stationary'
        self.movement_vector = (0, 0)
        self.has_crossed_stop_line = False

    @staticmethod
    def _calc_centroid(bbox: List[int]) -> Tuple[int, int]:
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def update(self, bbox: List[int], conf: float):
        self.bbox = [int(v) for v in bbox]
        self.conf = conf
        new_centroid = self._calc_centroid(self.bbox)

        # Compute movement vector
        dx = new_centroid[0] - self.centroid[0]
        dy = new_centroid[1] - self.centroid[1]
        self.movement_vector = (dx, dy)

        if abs(dy) > abs(dx) and abs(dy) > 2:
            self.direction = "downward" if dy > 0 else "upward"
        elif abs(dx) > abs(dy) and abs(dx) > 2:
            self.direction = "right" if dx > 0 else "left"

        self.centroid = new_centroid
        self.history.append(self.centroid)
        if len(self.history) > 30:
            self.history.pop(0)

        self.disappeared = 0
        self.frames_active += 1

    def has_violation(self, violation_type: str) -> bool:
        return violation_type in self.violations_flagged

    def mark_violation(self, violation_type: str):
        self.violations_flagged.add(violation_type)


class ObjectTracker:
    """Multi-object tracker associating detections across video frames."""

    def __init__(self, max_disappeared: int = 15, max_distance: float = 80.0):
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.next_id = 101
        self.tracks: Dict[str, TrackedObject] = {}

    def update(self, detections: List[Dict[str, Any]]) -> List[TrackedObject]:
        """
        Updates tracker state with new frame detections.
        detections: list of dicts with 'bbox', 'class_name', 'confidence'.
        """
        if len(detections) == 0:
            for tid in list(self.tracks.keys()):
                self.tracks[tid].disappeared += 1
                if self.tracks[tid].disappeared > self.max_disappeared:
                    del self.tracks[tid]
            return list(self.tracks.values())

        det_centroids = []
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            det_centroids.append(((x1 + x2) // 2, (y1 + y2) // 2))

        # First run: register all detections
        if len(self.tracks) == 0:
            for d in detections:
                tid = f"TRK-{self.next_id}"
                self.next_id += 1
                self.tracks[tid] = TrackedObject(tid, d["class_name"], d["bbox"], d["confidence"])
            return list(self.tracks.values())

        track_ids = list(self.tracks.keys())
        track_centroids = [self.tracks[tid].centroid for tid in track_ids]

        # Centroid distance matrix
        D = np.zeros((len(track_centroids), len(det_centroids)), dtype=np.float32)
        for i, tc in enumerate(track_centroids):
            for j, dc in enumerate(det_centroids):
                D[i, j] = np.linalg.norm(np.array(tc) - np.array(dc))

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for r, c in zip(rows, cols):
            if r in used_rows or c in used_cols:
                continue
            if D[r, c] > self.max_distance:
                continue

            tid = track_ids[r]
            det = detections[c]
            self.tracks[tid].update(det["bbox"], det["confidence"])
            self.tracks[tid].class_name = det["class_name"]

            used_rows.add(r)
            used_cols.add(c)

        # Handle disappeared
        unused_rows = set(range(len(track_ids))) - used_rows
        for r in unused_rows:
            tid = track_ids[r]
            self.tracks[tid].disappeared += 1
            if self.tracks[tid].disappeared > self.max_disappeared:
                del self.tracks[tid]

        # Handle new tracks
        unused_cols = set(range(len(detections))) - used_cols
        for c in unused_cols:
            det = detections[c]
            tid = f"TRK-{self.next_id}"
            self.next_id += 1
            self.tracks[tid] = TrackedObject(tid, det["class_name"], det["bbox"], det["confidence"])

        return list(self.tracks.values())

    def reset(self):
        """Resets all active tracking states."""
        self.tracks.clear()
        self.next_id = 101
