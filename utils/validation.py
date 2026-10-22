"""
Vision Guard — Validation & Formatting Utilities
Input validation, security sanitization, plate format checks, and presentation formatting.
"""

import re
import os
from typing import Tuple, List

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB limit


def validate_file(filename: str, file_bytes: bytes, is_video: bool = False) -> Tuple[bool, str]:
    """Validates uploaded file size and extension."""
    if not filename:
        return False, "Filename cannot be empty."

    _, ext = os.path.splitext(filename.lower())
    allowed = ALLOWED_VIDEO_EXTENSIONS if is_video else ALLOWED_IMAGE_EXTENSIONS
    if ext not in allowed:
        return False, f"Unsupported file format '{ext}'. Allowed: {', '.join(sorted(allowed))}"

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        max_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
        return False, f"File exceeds maximum upload size limit of {max_mb:.0f} MB."

    if len(file_bytes) == 0:
        return False, "Uploaded file contains 0 bytes."

    return True, "Valid"


def sanitize_filename(filename: str) -> str:
    """Removes path traversals and illegal filesystem characters."""
    base = os.path.basename(filename)
    sanitized = re.sub(r'[^a-zA-Z0-9_.-]', '_', base)
    return sanitized or "upload.bin"


def normalize_license_plate(plate_raw: str) -> str:
    """
    Cleans raw OCR text: strips noise, capitalizes, and standardizes spacing.
    Supports standard formats like DL01AB1234 -> DL 01 AB 1234 or clean international tokens.
    """
    if not plate_raw:
        return "UNREADABLE"

    # Remove non-alphanumeric except spaces and hyphens
    cleaned = re.sub(r'[^A-Za-z0-9]', '', plate_raw).upper()
    if len(cleaned) < 4:
        return "UNREADABLE"

    # Standard Indian format heuristic: 2 letters, 2 numbers, optional 1-2 letters, 4 numbers
    match = re.match(r'^([A-Z]{2})(\d{1,2})([A-Z]{0,3})(\d{1,4})$', cleaned)
    if match:
        parts = [g for g in match.groups() if g]
        return " ".join(parts)

    return cleaned


def format_currency(amount: float, symbol: str = "₹") -> str:
    """Formats monetary amounts with currency symbol and decimal formatting."""
    return f"{symbol}{amount:,.2f}"
