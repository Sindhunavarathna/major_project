"""Unit tests for image preprocessing and line region merging."""

import numpy as np

from src.preprocessing import (
    adaptive_threshold,
    denoise_image,
    deskew_image,
    enhance_contrast_mild,
    filter_noise_boxes,
    merge_word_boxes_into_lines,
    sauvola_threshold,
    segment_lines_morphological,
    segment_words_morphological,
    AblationConfig,
    run_ablation_pipeline,
    to_grayscale,
)


def test_to_grayscale(synthetic_text_image: np.ndarray):
    gray = to_grayscale(synthetic_text_image)
    assert gray.ndim == 2
    assert gray.shape[:2] == synthetic_text_image.shape[:2]

    gray2 = to_grayscale(gray)
    assert gray2.ndim == 2
    assert np.array_equal(gray, gray2)

    rgba = np.dstack([synthetic_text_image, np.full((120, 400), 255, dtype=np.uint8)])
    gray_rgba = to_grayscale(rgba)
    assert gray_rgba.ndim == 2


def test_denoise_image(synthetic_text_image: np.ndarray):
    gray = to_grayscale(synthetic_text_image)
    denoised_nl = denoise_image(gray, method="fastNlMeans", strength=8)
    assert denoised_nl.shape == gray.shape

    denoised_bi = denoise_image(gray, method="bilateral", strength=5)
    assert denoised_bi.shape == gray.shape

    denoised_ga = denoise_image(gray, method="gaussian", strength=6)
    assert denoised_ga.shape == gray.shape


def test_adaptive_threshold(synthetic_text_image: np.ndarray):
    gray = to_grayscale(synthetic_text_image)
    otsu = adaptive_threshold(gray, method="otsu")
    assert otsu.shape == gray.shape
    unique_vals = set(np.unique(otsu))
    assert unique_vals.issubset({0, 255})

    adapt = adaptive_threshold(gray, method="adaptive", block_size=15, c=4)
    assert adapt.shape == gray.shape
    assert set(np.unique(adapt)).issubset({0, 255})


def test_deskew_image(synthetic_skewed_image):
    skewed, true_angle = synthetic_skewed_image
    deskewed, detected_angle = deskew_image(skewed, max_angle=45.0)

    assert deskewed is not None
    assert deskewed.ndim == 3
    assert abs(detected_angle) > 1.0


def test_filter_noise_boxes():
    boxes = [
        (10, 10, 200, 40, 0.95, "Valid manuscript line"),
        (5, 5, 12, 12, 0.20, "# 0"),  # Noise symbol box
        (50, 50, 52, 53, 0.10, "."),   # Tiny dot
    ]
    filtered = filter_noise_boxes(boxes, min_width=20, min_height=10, min_area=200, min_conf=0.25)
    assert len(filtered) == 1
    assert filtered[0][5] == "Valid manuscript line"


def test_merge_word_boxes_into_lines():
    # Two words on line 1, one word on line 2
    boxes = [
        (10, 20, 50, 45, 0.90, "Hello"),
        (60, 22, 110, 46, 0.92, "World"),
        (12, 70, 90, 95, 0.88, "Second"),
    ]
    merged = merge_word_boxes_into_lines(boxes, image_shape=(120, 400), padding_px=4)
    assert len(merged) == 2
    # Line 1 should merge "Hello" and "World"
    assert merged[0][5] == ["Hello", "World"]
    assert merged[1][5] == ["Second"]


def test_enhance_contrast_mild(synthetic_text_image: np.ndarray):
    enhanced = enhance_contrast_mild(synthetic_text_image)
    assert enhanced.shape == synthetic_text_image.shape
    assert enhanced.dtype == np.uint8


def test_sauvola_threshold(synthetic_text_image: np.ndarray):
    gray = to_grayscale(synthetic_text_image)
    sauv = sauvola_threshold(gray, window_size=25, k=0.2)
    assert sauv.shape == gray.shape
    assert set(np.unique(sauv)).issubset({0, 255})

    # Test via adaptive_threshold interface
    sauv2 = adaptive_threshold(gray, method="sauvola", block_size=25, sauvola_k=0.2)
    assert np.array_equal(sauv, sauv2)


def test_morphological_segmentation(synthetic_text_image: np.ndarray):
    lines = segment_lines_morphological(synthetic_text_image, min_line_height=8, min_line_width=20)
    assert len(lines) >= 1
    for line in lines:
        assert line.crop.size > 0
        words = segment_words_morphological(line.crop, min_word_width=4, min_word_height=4)
        assert isinstance(words, list)


def test_ablation_pipeline(synthetic_text_image: np.ndarray):
    cfg = AblationConfig(
        grayscale=True,
        denoise=True,
        denoise_method="bilateral",
        deskew=True,
        binarize=True,
        binarize_method="sauvola",
        segment_lines=True,
        segment_words=True,
    )
    result = run_ablation_pipeline(synthetic_text_image, config=cfg)
    assert result.output_image is not None
    assert "grayscale" in result.stages
    assert "denoised" in result.stages
    assert "deskewed" in result.stages
    assert "binarized" in result.stages
    assert "denoise" in result.timings_ms
    assert len(result.lines) >= 1
    assert "line_count" in result.metrics

