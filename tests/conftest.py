"""Pytest configuration and shared fixtures."""

from pathlib import Path
import cv2
import numpy as np
import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def synthetic_text_image() -> np.ndarray:
    """Create a 400x120 clean BGR image with black text on white background."""
    img = np.ones((120, 400, 3), dtype=np.uint8) * 255
    cv2.putText(
        img,
        "ANCIENT MANUSCRIPT",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return img


@pytest.fixture
def synthetic_multiline_image() -> np.ndarray:
    """Create a 400x200 multiline image for line segmentation testing."""
    img = np.ones((200, 400, 3), dtype=np.uint8) * 255
    cv2.putText(
        img,
        "Line Number One",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "Line Number Two",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "Line Number Three",
        (20, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return img


@pytest.fixture
def synthetic_skewed_image(synthetic_text_image: np.ndarray) -> tuple[np.ndarray, float]:
    """Rotate image by 10 degrees to test deskew."""
    angle = 10.0
    h, w = synthetic_text_image.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        synthetic_text_image,
        m,
        (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return rotated, angle

