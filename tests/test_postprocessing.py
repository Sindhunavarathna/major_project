"""Unit tests for postprocessing module and visual document translation."""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.postprocessing import (
    create_translated_image,
    fit_text_lines_to_box,
    inpaint_text_region,
    sample_background_color,
)
from src.preprocessing import TextLineRegion


def test_sample_background_color():
    # Create image with specific background color (e.g. 240, 230, 220)
    bg_val = (240, 230, 220)
    img = np.full((100, 200, 3), bg_val, dtype=np.uint8)

    color = sample_background_color(img, (20, 20, 180, 80))
    assert color == bg_val


def test_inpaint_text_region(synthetic_text_image: np.ndarray):
    bbox = (20, 20, 200, 80)
    clean = inpaint_text_region(synthetic_text_image.copy(), bbox, use_telea=True)
    assert clean.shape == synthetic_text_image.shape
    assert clean.dtype == np.uint8


def test_fit_text_lines_to_box():
    dummy = Image.new("RGB", (300, 100))
    draw = ImageDraw.Draw(dummy)

    font, lines = fit_text_lines_to_box(
        draw=draw,
        text="Ancient manuscripts preserve knowledge across civilizations.",
        max_w=200,
        max_h=60,
    )
    assert font is not None
    assert len(lines) >= 1
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        assert (bbox[2] - bbox[0]) <= 220


def test_create_translated_image(synthetic_text_image: np.ndarray):
    regions = [
        TextLineRegion(
            bbox_xyxy=(20, 30, 350, 90),
            confidence=0.92,
            word_count=2,
            crop=synthetic_text_image[30:90, 20:350],
        )
    ]
    translated_text = "Ancient manuscript in English translation."

    res = create_translated_image(
        image=synthetic_text_image,
        line_regions=regions,
        translated_text=translated_text,
    )

    assert res is not None
    assert res.shape == synthetic_text_image.shape
    assert res.dtype == np.uint8

