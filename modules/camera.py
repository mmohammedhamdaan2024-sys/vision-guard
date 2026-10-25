"""
VisionGuard — Camera Module
Robust OpenCV video capture handling for live webcams and video stream processing.
Includes camera lifecycle management, FPS calculation, pause/resume, and device scanning.
"""

import time
import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict, Any


class CameraManager:
    """Manages OpenCV VideoCapture sessions with clean initialization and release."""

    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.is_paused = False

        # FPS metrics
        self.prev_frame_time = 0.0
        self.fps = 0.0
        self.frame_count = 0

    @staticmethod
    def get_available_cameras(max_tested: int = 3) -> List[int]:
        """Scans camera indices and returns a list of working camera device indices."""
        available = []
        for idx in range(max_tested):
            temp_cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if cv2.os.name == 'nt' else cv2.VideoCapture(idx)
            if temp_cap.isOpened():
                ret, _ = temp_cap.read()
                if ret:
                    available.append(idx)
                temp_cap.release()
        return available

    def start(self, source_path_or_idx: Optional[Any] = None) -> Tuple[bool, str]:
        """
        Initializes and opens the camera or video source.
        source_path_or_idx: int (camera index) or str (video file path).
        """
        self.stop()  # Clean up existing capture if any

        target = source_path_or_idx if source_path_or_idx is not None else self.camera_index

        try:
            if isinstance(target, int):
                # Prefer DirectShow on Windows for fast webcam init
                self.cap = cv2.VideoCapture(target, cv2.CAP_DSHOW) if cv2.os.name == 'nt' else cv2.VideoCapture(target)
                if not self.cap.isOpened():
                    # Fallback to standard backend
                    self.cap = cv2.VideoCapture(target)
            else:
                self.cap = cv2.VideoCapture(str(target))

            if not self.cap or not self.cap.isOpened():
                return False, f"Could not connect to camera / video source '{target}'."

            # Set preferred resolution
            if isinstance(target, int):
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

            self.is_running = True
            self.is_paused = False
            self.prev_frame_time = time.time()
            self.frame_count = 0
            return True, "Camera connected successfully."

        except Exception as e:
            self.is_running = False
            return False, f"Camera error: {str(e)}"

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Reads a frame from the capture stream and calculates real-time FPS."""
        if not self.is_running or self.cap is None:
            return False, None

        if self.is_paused:
            return True, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None

        # Resize if dimensions differ significantly
        if self.width and self.height:
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        # Calculate FPS
        curr_time = time.time()
        time_diff = curr_time - self.prev_frame_time
        if time_diff > 0:
            instant_fps = 1.0 / time_diff
            self.fps = 0.85 * self.fps + 0.15 * instant_fps if self.fps > 0 else instant_fps
        self.prev_frame_time = curr_time
        self.frame_count += 1

        return True, frame

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False
        self.prev_frame_time = time.time()

    def stop(self):
        """Safely stops and releases the camera resource."""
        self.is_running = False
        self.is_paused = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.fps = 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Returns runtime camera status and performance indicators."""
        return {
            "is_running": self.is_running,
            "is_paused": self.is_paused,
            "fps": round(self.fps, 1),
            "frame_count": self.frame_count,
            "camera_index": self.camera_index,
            "resolution": f"{self.width}x{self.height}"
        }
