"""
Vision Guard — Vehicle Tracking & Deduplication Module
Centroid and IoU tracker maintaining temporal consistency, vehicle IDs,
trajectory histories, and preventing duplicate violation triggers.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional


class TrackedVehicle:
    def __init__(self, track_id: str, vehicle_type: str, bbox: List[float], conf: float):
        self.track_id = track_id
        self.vehicle_type = vehicle_type
        self.bbox = [int(v) for v in bbox]
        self.conf = conf
        self.centroid = self._calc_centroid(self.bbox)
        self.history = [self.centroid]
        self.disappeared = 0
        self.violations_flagged = set()
        self.plate_number = "UNREADABLE"
        self.plate_conf = 0.0
        self.frames_active = 1

    @staticmethod
    def _calc_centroid(bbox: List[int]) -> Tuple[int, int]:
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def update(self, bbox: List[float], conf: float):
        self.bbox = [int(v) for v in bbox]
        self.conf = conf
        self.centroid = self._calc_centroid(self.bbox)
        self.history.append(self.centroid)
        if len(self.history) > 30:
            self.history.pop(0)
        self.disappeared = 0
        self.frames_active += 1

    def has_violation_flagged(self, violation_type: str) -> bool:
        return violation_type in self.violations_flagged

    def mark_violation_flagged(self, violation_type: str):
        self.violations_flagged.add(violation_type)


class VehicleTracker:
    def __init__(self, max_disappeared: int = 15, max_distance: float = 85.0):
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.next_id = 101
        self.tracks: Dict[str, TrackedVehicle] = {}

    def update(self, detections: List[Tuple[List[float], str, float]]) -> List[TrackedVehicle]:
        """
        Updates tracking state with detections: list of (bbox, vehicle_type, confidence).
        Matches existing tracks using minimum Euclidean centroid distance.
        """
        if len(detections) == 0:
            # Mark all tracks as disappeared
            for tid in list(self.tracks.keys()):
                self.tracks[tid].disappeared += 1
                if self.tracks[tid].disappeared > self.max_disappeared:
                    del self.tracks[tid]
            return list(self.tracks.values())

        det_centroids = []
        for box, _, _ in detections:
            x1, y1, x2, y2 = [int(v) for v in box]
            det_centroids.append(((x1 + x2) // 2, (y1 + y2) // 2))

        # If no active tracks, register all detections as new
        if len(self.tracks) == 0:
            for (box, vtype, conf) in detections:
                tid = f"TRK-{self.next_id}"
                self.next_id += 1
                self.tracks[tid] = TrackedVehicle(tid, vtype, box, conf)
            return list(self.tracks.values())

        # Match existing tracks with detections
        track_ids = list(self.tracks.keys())
        track_centroids = [self.tracks[tid].centroid for tid in track_ids]

        # Distance matrix
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
            box, vtype, conf = detections[c]
            self.tracks[tid].update(box, conf)
            self.tracks[tid].vehicle_type = vtype

            used_rows.add(r)
            used_cols.add(c)

        # Unmatched tracks
        unused_rows = set(range(len(track_ids))) - used_rows
        for r in unused_rows:
            tid = track_ids[r]
            self.tracks[tid].disappeared += 1
            if self.tracks[tid].disappeared > self.max_disappeared:
                del self.tracks[tid]

        # New detections
        unused_cols = set(range(len(detections))) - used_cols
        for c in unused_cols:
            box, vtype, conf = detections[c]
            tid = f"TRK-{self.next_id}"
            self.next_id += 1
            self.tracks[tid] = TrackedVehicle(tid, vtype, box, conf)

        return list(self.tracks.values())
