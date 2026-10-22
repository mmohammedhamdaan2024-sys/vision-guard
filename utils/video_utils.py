"""
Vision Guard — Video Utilities
Frame processing, metadata extraction, and synthetic traffic video generation for live simulation.
"""

import os
import cv2
import numpy as np
from typing import Generator, Tuple, Dict, Any


def get_video_metadata(video_path: str) -> Dict[str, Any]:
    """Retrieves video duration, frame rate, total frames, and dimensions."""
    if not os.path.exists(video_path):
        return {"error": "File does not exist", "fps": 0, "total_frames": 0, "width": 0, "height": 0}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Cannot open video stream", "fps": 0, "total_frames": 0, "width": 0, "height": 0}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    duration_sec = total_frames / fps if fps > 0 else 0

    cap.release()
    return {
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration_sec": round(duration_sec, 2)
    }


def frame_generator(video_path: str, frame_skip: int = 2,
                    target_size: Tuple[int, int] = (854, 480)) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Yields sampled video frames with resizing for performant web dashboard rendering.
    """
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            if target_size and (frame.shape[1] != target_size[0] or frame.shape[0] != target_size[1]):
                frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
            yield frame_idx, frame

        frame_idx += 1

    cap.release()


def generate_synthetic_traffic_clip(output_path: str, num_frames: int = 75, fps: int = 15) -> str:
    """
    Generates a realistic 5-second MP4 traffic simulation clip featuring moving
    vehicles, motorcycle rider, and lane dividers. Ensures immediate demo capability.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    width, height = 854, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for f in range(num_frames):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        # Asphalt
        canvas[:] = (40, 44, 52)

        # Highway lane markings
        offset = (f * 8) % 100
        for y in range(-50, height + 100, 100):
            # Center lane dashed yellow
            cv2.rectangle(canvas, (width // 2 - 4, y + offset), (width // 2 + 4, y + offset + 50), (0, 215, 255), -1)
            # Left lane dashed white
            cv2.rectangle(canvas, (width // 4 - 3, y + offset), (width // 4 + 3, y + offset + 50), (240, 240, 240), -1)
            # Right lane dashed white
            cv2.rectangle(canvas, (3 * width // 4 - 3, y + offset), (3 * width // 4 + 3, y + offset + 50), (240, 240, 240), -1)

        # Vehicle 1: Car moving down in Left-Center Lane
        car1_y = int(50 + (f * 4.2))
        car1_x = 240
        if car1_y < height + 80:
            cv2.rectangle(canvas, (car1_x, car1_y), (car1_x + 120, car1_y + 190), (30, 40, 160), -1) # Blue car body
            cv2.rectangle(canvas, (car1_x + 15, car1_y + 40), (car1_x + 105, car1_y + 100), (140, 160, 170), -1) # Windshield
            # License plate
            cv2.rectangle(canvas, (car1_x + 30, car1_y + 165), (car1_x + 90, car1_y + 185), (255, 255, 255), -1)
            cv2.putText(canvas, "MH12AB", (car1_x + 32, car1_y + 180), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

        # Vehicle 2: Motorcycle moving down in Right Lane (No Helmet Rider)
        bike_y = int(20 + (f * 5.5))
        bike_x = 580
        if bike_y < height + 60:
            # Bike chassis
            cv2.rectangle(canvas, (bike_x + 15, bike_y + 40), (bike_x + 35, bike_y + 110), (20, 20, 20), -1)
            # Rider body
            cv2.rectangle(canvas, (bike_x + 12, bike_y + 25), (bike_x + 38, bike_y + 65), (50, 120, 220), -1)
            # Rider head (No helmet: exposed skin & dark hair)
            cv2.circle(canvas, (bike_x + 25, bike_y + 12), 12, (130, 175, 225), -1) # Skin tone
            cv2.circle(canvas, (bike_x + 25, bike_y + 6), 9, (20, 20, 20), -1) # Hair
            # Bike small plate
            cv2.rectangle(canvas, (bike_x + 16, bike_y + 100), (bike_x + 34, bike_y + 112), (255, 255, 255), -1)

        # Vehicle 3: White SUV in Right-Center Lane
        suv_y = int(-100 + (f * 3.8))
        suv_x = 390
        if -50 <= suv_y < height + 100:
            cv2.rectangle(canvas, (suv_x, suv_y), (suv_x + 130, suv_y + 210), (220, 220, 220), -1) # White SUV
            cv2.rectangle(canvas, (suv_x + 20, suv_y + 40), (suv_x + 110, suv_y + 90), (30, 30, 30), -1) # Dark tinted front windshield!
            cv2.rectangle(canvas, (suv_x + 35, suv_y + 185), (suv_x + 95, suv_y + 205), (255, 255, 255), -1)
            cv2.putText(canvas, "DL04XY", (suv_x + 37, suv_y + 200), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

        # Surveillance camera HUD header overlay
        cv2.putText(canvas, f"CAM-04 [HIGHWAY-EXPRESSWAY] | FRAME: {f:04d} | TIME: 19:45:{f%60:02d}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(canvas, (width - 40, 25), 8, (0, 0, 255), -1) # Blinking REC dot
        cv2.putText(canvas, "REC", (width - 75, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        out.write(canvas)

    out.release()
    return output_path
