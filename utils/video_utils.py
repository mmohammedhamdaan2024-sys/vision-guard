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
        canvas[:] = (38, 42, 48)

        # Stop Line across all lanes
        stop_line_y = 330
        cv2.rectangle(canvas, (40, stop_line_y), (width - 120, stop_line_y + 8), (255, 255, 255), -1)
        # Zebra Crosswalk below stop line
        for zx in range(60, width - 140, 45):
            cv2.rectangle(canvas, (zx, stop_line_y + 16), (zx + 25, stop_line_y + 55), (230, 230, 230), -1)

        # Lane dividers
        for lx in [width // 3, 2 * width // 3]:
            for y in range(0, stop_line_y, 40):
                cv2.rectangle(canvas, (lx - 2, y), (lx + 2, y + 25), (240, 240, 240), -1)

        # Vehicle 1: Blue Car in Left Lane (Slows and stops at traffic signal)
        car1_prog = min(1.0, f / (num_frames * 0.75))
        car1_y = int(30 + car1_prog * 120)
        car1_x = 110
        # Car body
        cv2.rectangle(canvas, (car1_x, car1_y), (car1_x + 130, car1_y + 160), (160, 40, 30), -1) # Deep blue
        # Windshield
        cv2.rectangle(canvas, (car1_x + 15, car1_y + 35), (car1_x + 115, car1_y + 85), (195, 215, 225), -1)
        # Driver Head & Visible Fastened Diagonal Seatbelt
        cv2.circle(canvas, (car1_x + 45, car1_y + 55), 14, (120, 160, 210), -1)
        # Fastened diagonal black seat-belt strap across torso!
        cv2.line(canvas, (car1_x + 35, car1_y + 50), (car1_x + 65, car1_y + 85), (20, 20, 20), 4)
        # Passenger (No belt across)
        cv2.circle(canvas, (car1_x + 85, car1_y + 55), 14, (120, 160, 210), -1)
        # License plate
        cv2.rectangle(canvas, (car1_x + 30, car1_y + 140), (car1_x + 100, car1_y + 155), (255, 255, 255), -1)
        cv2.putText(canvas, "MH12AB1234", (car1_x + 32, car1_y + 152), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1)

        # Vehicle 2: Motorcycle Rider in Center Lane (No Helmet Rider)
        bike_prog = min(1.0, f / (num_frames * 0.7))
        bike_y = int(20 + bike_prog * 140)
        bike_x = 360
        # Motorcycle chassis
        cv2.rectangle(canvas, (bike_x + 15, bike_y + 40), (bike_x + 35, bike_y + 115), (25, 25, 25), -1)
        cv2.circle(canvas, (bike_x + 25, bike_y + 115), 12, (10, 10, 10), -1) # Wheel
        # Rider torso
        cv2.rectangle(canvas, (bike_x + 10, bike_y + 20), (bike_x + 40, bike_y + 65), (50, 120, 220), -1)
        # Rider Head (NO HELMET: Exposed skin tone and dark hair)
        cv2.circle(canvas, (bike_x + 25, bike_y + 10), 13, (120, 160, 210), -1) # Face skin
        cv2.ellipse(canvas, (bike_x + 25, bike_y + 4), (12, 7), 0, 0, 180, (20, 20, 20), -1) # Hair
        # Number plate
        cv2.rectangle(canvas, (bike_x + 14, bike_y + 102), (bike_x + 36, bike_y + 114), (255, 255, 255), -1)
        cv2.putText(canvas, "DL03", (bike_x + 15, bike_y + 112), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 0, 0), 1)

        # Vehicle 3: White SUV in Right Lane with dark window film
        suv_prog = min(1.0, f / (num_frames * 0.8))
        suv_y = int(-10 + suv_prog * 125)
        suv_x = 475
        cv2.rectangle(canvas, (suv_x, suv_y), (suv_x + 135, suv_y + 175), (225, 225, 225), -1)
        # Heavy dark tinted windshield
        cv2.rectangle(canvas, (suv_x + 18, suv_y + 35), (suv_x + 118, suv_y + 85), (20, 20, 20), -1)
        # SUV plate
        cv2.rectangle(canvas, (suv_x + 35, suv_y + 152), (suv_x + 100, suv_y + 168), (255, 255, 255), -1)
        cv2.putText(canvas, "KA05MN9010", (suv_x + 37, suv_y + 165), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1)

        # Traffic Signal Light Pole on the Right
        pole_x = width - 80
        cv2.rectangle(canvas, (pole_x + 10, 80), (pole_x + 18, height), (70, 75, 80), -1) # Pole
        # Signal Box
        cv2.rectangle(canvas, (pole_x - 5, 80), (pole_x + 35, 195), (20, 22, 25), -1)
        cv2.rectangle(canvas, (pole_x - 5, 80), (pole_x + 35, 195), (100, 105, 110), 2)
        # Traffic Lights (Red Glowing ON, Yellow Dim, Green Dim)
        cv2.circle(canvas, (pole_x + 15, 105), 12, (0, 0, 255), -1) # RED LIGHT ON!
        cv2.circle(canvas, (pole_x + 15, 105), 15, (0, 0, 255), 1)
        cv2.circle(canvas, (pole_x + 15, 137), 10, (30, 45, 50), -1) # Yellow off
        cv2.circle(canvas, (pole_x + 15, 168), 10, (20, 40, 20), -1) # Green off
        cv2.putText(canvas, "SIGNAL: RED", (pole_x - 30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)

        # Surveillance camera HUD header overlay
        cv2.putText(canvas, f"TRAFFIC SIGNAL CAM-01 [INTERSECTION JUNCTION] | FRAME: {f:04d}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "ROAD SAFETY AI ACTIVE: SCANNING PASSENGER SEATBELT & RIDER HELMET", (20, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 220, 255), 1, cv2.LINE_AA)
        cv2.circle(canvas, (width - 40, 25), 8, (0, 0, 255), -1) # Blinking REC dot
        cv2.putText(canvas, "REC", (width - 75, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        out.write(canvas)

    out.release()
    return output_path
