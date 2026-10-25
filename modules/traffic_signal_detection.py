"""
VisionGuard — Traffic Signal & Stop-Line Violation Detection Module
Monitors traffic signal states (Red, Yellow, Green) and detects vehicles crossing
a user-configurable stop line during active red light intervals.
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from config import COLORS, DEFAULT_STOP_LINE_Y_RATIO


class TrafficSignalMonitor:
    """Detects traffic light states and tracks stop-line crossing infractions."""

    def __init__(self, stop_line_y_ratio: float = DEFAULT_STOP_LINE_Y_RATIO,
                 manual_signal_state: str = "RED"):
        self.stop_line_y_ratio = stop_line_y_ratio
        self.current_signal_state = manual_signal_state.upper()  # 'RED', 'YELLOW', 'GREEN'
        self.detected_light_bbox: Optional[List[int]] = None
        self.signal_confidence = 0.85

    def set_signal_state(self, state: str):
        """Allows user or automated detector to update current light state."""
        state_upper = state.upper()
        if state_upper in ("RED", "YELLOW", "GREEN"):
            self.current_signal_state = state_upper

    def set_stop_line_ratio(self, ratio: float):
        """Adjusts the vertical position of the virtual stop line (0.1 to 0.9)."""
        self.stop_line_y_ratio = max(0.1, min(0.95, ratio))

    def detect_traffic_light_color(self, frame: np.ndarray, light_box: Optional[List[int]] = None) -> str:
        """
        Analyzes HSV color dominance inside candidate traffic light bounding box.
        Returns 'RED', 'YELLOW', 'GREEN', or 'UNKNOWN'.
        """
        if frame is None:
            return "UNKNOWN"

        h, w = frame.shape[:2]
        if light_box is not None:
            lx1, ly1, lx2, ly2 = [int(v) for v in light_box]
            lx1, ly1 = max(0, lx1), max(0, ly1)
            lx2, ly2 = min(w, lx2), min(h, ly2)
            light_crop = frame[ly1:ly2, lx1:lx2]
        else:
            # Look in upper 40% right or left corner of frame for traffic light
            light_crop = frame[0:int(h * 0.45), int(w * 0.65):w]

        if light_crop is None or light_crop.size == 0:
            return self.current_signal_state

        hsv = cv2.cvtColor(light_crop, cv2.COLOR_BGR2HSV)

        # Red mask (wraps around 0 and 180)
        red_lower1 = np.array([0, 100, 100], dtype=np.uint8)
        red_upper1 = np.array([10, 255, 255], dtype=np.uint8)
        red_lower2 = np.array([160, 100, 100], dtype=np.uint8)
        red_upper2 = np.array([180, 255, 255], dtype=np.uint8)
        mask_red = cv2.inRange(hsv, red_lower1, red_upper1) | cv2.inRange(hsv, red_lower2, red_upper2)

        # Yellow mask
        yellow_lower = np.array([15, 100, 100], dtype=np.uint8)
        yellow_upper = np.array([35, 255, 255], dtype=np.uint8)
        mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)

        # Green mask
        green_lower = np.array([40, 100, 100], dtype=np.uint8)
        green_upper = np.array([90, 255, 255], dtype=np.uint8)
        mask_green = cv2.inRange(hsv, green_lower, green_upper)

        red_pixels = int(np.sum(mask_red > 0))
        yellow_pixels = int(np.sum(mask_yellow > 0))
        green_pixels = int(np.sum(mask_green > 0))

        if red_pixels > 25 and red_pixels > yellow_pixels and red_pixels > green_pixels:
            self.current_signal_state = "RED"
        elif green_pixels > 25 and green_pixels > red_pixels and green_pixels > yellow_pixels:
            self.current_signal_state = "GREEN"
        elif yellow_pixels > 25:
            self.current_signal_state = "YELLOW"

        return self.current_signal_state

    def check_stop_line_crossing(self, frame_height: int,
                                 prev_centroid: Tuple[int, int],
                                 curr_centroid: Tuple[int, int],
                                 vehicle_bbox: List[int]) -> Tuple[bool, Dict[str, Any]]:
        """
        Determines if a moving vehicle crossed the configured stop line.
        Crossing occurs when previous bottom/centroid was above the line,
        and current bottom/centroid has crossed over it while moving forward.
        """
        stop_line_y = int(frame_height * self.stop_line_y_ratio)

        # Use vehicle bottom edge (front bumper when moving downward)
        prev_y = prev_centroid[1]
        curr_y = curr_centroid[1]
        bottom_y = vehicle_bbox[3]

        # Vehicle is moving downward towards the intersection
        has_crossed = (prev_y < stop_line_y and bottom_y >= stop_line_y) or (
            prev_y <= stop_line_y and curr_y > stop_line_y
        )

        is_red_light_violation = has_crossed and (self.current_signal_state == "RED")

        details = {
            "stop_line_y": stop_line_y,
            "signal_state": self.current_signal_state,
            "has_crossed": has_crossed,
            "is_violation": is_red_light_violation,
            "direction": "downward" if curr_y > prev_y else "upward"
        }

        return is_red_light_violation, details

    def draw_hud(self, frame: np.ndarray) -> np.ndarray:
        """Renders the virtual stop line, signal state badge, and intersection HUD."""
        out = frame.copy()
        h, w = out.shape[:2]
        stop_line_y = int(h * self.stop_line_y_ratio)

        # Stop line color based on signal state
        line_color = COLORS["stop_line_red"] if self.current_signal_state == "RED" else (
            (0, 200, 255) if self.current_signal_state == "YELLOW" else COLORS["stop_line_green"]
        )

        # Draw solid virtual stop line
        cv2.line(out, (0, stop_line_y), (w, stop_line_y), line_color, 3)

        # Stop Line Tag
        tag_text = f"STOP LINE [SIGNAL: {self.current_signal_state}]"
        cv2.putText(out, tag_text, (20, stop_line_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, line_color, 2, cv2.LINE_AA)

        # Draw Signal Status Box on Top Right
        box_w, box_h = 175, 45
        cv2.rectangle(out, (w - box_w - 15, 15), (w - 15, 15 + box_h), (20, 24, 34), -1)
        cv2.rectangle(out, (w - box_w - 15, 15), (w - 15, 15 + box_h), line_color, 2)

        # Signal circle icon
        circle_x = w - box_w + 5
        circle_y = 15 + (box_h // 2)
        cv2.circle(out, (circle_x, circle_y), 11, line_color, -1)

        cv2.putText(out, f"LIGHT: {self.current_signal_state}", (circle_x + 22, circle_y + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        return out
