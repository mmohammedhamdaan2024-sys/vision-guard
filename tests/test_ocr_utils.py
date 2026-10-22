"""
Vision Guard — Unit Tests for OCR Normalization and File Validation
"""

from utils.validation import normalize_license_plate, validate_file, sanitize_filename


def test_plate_normalization():
    # Clean standard formats
    assert normalize_license_plate("DL01AB1234") == "DL 01 AB 1234"
    assert normalize_license_plate("mh12cd5678") == "MH 12 CD 5678"
    assert normalize_license_plate("KA 05 MN 9010") == "KA 05 MN 9010"

    # Edge cases
    assert normalize_license_plate("") == "UNREADABLE"
    assert normalize_license_plate("??##") == "UNREADABLE"


def test_file_validation():
    valid_img = b"fakeimagedata"
    valid, msg = validate_file("test.jpg", valid_img, is_video=False)
    assert valid is True

    valid_vid, msg = validate_file("traffic.mp4", valid_img, is_video=True)
    assert valid_vid is True

    # Bad extension
    invalid, msg = validate_file("virus.exe", valid_img, is_video=False)
    assert invalid is False


def test_sanitize_filename():
    assert sanitize_filename("../../../etc/passwd") == "passwd"
    assert sanitize_filename("test file (1).png") == "test_file__1_.png"
