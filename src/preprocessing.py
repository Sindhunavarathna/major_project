"""Image preprocessing and text region line-merging module for historical manuscripts.

Includes:
- OpenCV image cleaning (deskew, denoise, mild contrast enhancement).
- Merging EasyOCR/detection word boxes into coherent horizontal text lines.
- Safe cropping from original-resolution images with padding.
- Noise filtering to eliminate isolated speckles and non-text artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.utils import get_logger, load_image

LOGGER = get_logger("preprocessing")


@dataclass
class TextLineRegion:
    """Represents a merged text line bounding box."""
    bbox_xyxy: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    word_count: int
    crop: np.ndarray


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an image to single-channel grayscale if not already."""
    if image is None:
        raise ValueError("Input image cannot be None.")
    if image.ndim == 2:
        return image.copy()
    if image.ndim == 3:
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unexpected image shape: {image.shape}")


def denoise_image(
    gray_image: np.ndarray,
    method: str = "fastNlMeans",
    strength: int = 10,
) -> np.ndarray:
    """Apply denoising to a grayscale manuscript image."""
    if gray_image is None or gray_image.ndim != 2:
        gray_image = to_grayscale(gray_image)

    strength = max(1, int(strength))

    if method == "fastNlMeans":
        return cv2.fastNlMeansDenoising(
            gray_image, None, h=float(strength), templateWindowSize=7, searchWindowSize=21
        )
    elif method == "bilateral":
        d = 9
        sigma = float(strength * 5)
        return cv2.bilateralFilter(gray_image, d=d, sigmaColor=sigma, sigmaSpace=sigma)
    elif method == "gaussian":
        ksize = 2 * (strength // 3) + 1
        ksize = max(3, ksize if ksize % 2 != 0 else ksize + 1)
        return cv2.GaussianBlur(gray_image, (ksize, ksize), 0)
    else:
        return cv2.fastNlMeansDenoising(
            gray_image, None, h=float(strength), templateWindowSize=7, searchWindowSize=21
        )


def sauvola_threshold(
    gray_image: np.ndarray,
    window_size: int = 25,
    k: float = 0.2,
    r: float = 128.0,
) -> np.ndarray:
    """Sauvola adaptive binarization for historical manuscripts and degraded documents.

    Threshold formula: T(x, y) = mean * (1 + k * (std / R - 1))
    Where R is the dynamic range of standard deviation (128 for 8-bit images).
    """
    if gray_image.ndim != 2:
        gray_image = to_grayscale(gray_image)

    w = window_size if window_size % 2 != 0 else window_size + 1
    w = max(3, w)

    img_f = gray_image.astype(np.float32)
    mean = cv2.blur(img_f, (w, w))
    sq_mean = cv2.blur(img_f ** 2, (w, w))
    variance = np.maximum(0.0, sq_mean - (mean ** 2))
    std = np.sqrt(variance)

    thresh = mean * (1.0 + k * ((std / r) - 1.0))
    binary = np.where(img_f >= thresh, 255, 0).astype(np.uint8)
    return binary


def adaptive_threshold(
    gray_image: np.ndarray,
    method: str = "otsu",
    block_size: int = 15,
    c: int = 4,
    sauvola_k: float = 0.2,
) -> np.ndarray:
    """Binarize manuscript image to separate ink text from parchment background.

    Supported methods:
    - 'otsu': Global Otsu optimal thresholding.
    - 'adaptive': Local Gaussian adaptive thresholding.
    - 'sauvola': Sauvola local adaptive binarization for historical documents.
    """
    if gray_image.ndim != 2:
        gray_image = to_grayscale(gray_image)

    if method.lower() == "sauvola":
        return sauvola_threshold(gray_image, window_size=block_size, k=sauvola_k)
    elif method.lower() == "adaptive":
        bs = block_size if block_size % 2 != 0 else block_size + 1
        bs = max(3, bs)
        binary = cv2.adaptiveThreshold(
            gray_image,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=bs,
            C=float(c),
        )
    else:  # Otsu
        _, binary = cv2.threshold(
            gray_image,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

    return binary


def enhance_contrast_mild(bgr_image: np.ndarray) -> np.ndarray:
    """Mild contrast enhancement for TrOCR input.

    Applies CLAHE on the luminance channel of LAB color space to preserve ink
    gradients and paper texture without harsh binarization artifacts.
    """
    if bgr_image.ndim == 2:
        bgr = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = bgr_image.copy()

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_bgr


def compute_skew_angle(binary_image: np.ndarray, max_angle: float = 45.0) -> float:
    """Calculate the dominant orientation skew angle in degrees."""
    text_pixels = np.column_stack(np.where(binary_image < 128))
    if len(text_pixels) < 50:
        return 0.0

    rect = cv2.minAreaRect(text_pixels)
    angle = rect[-1]

    if angle < -45.0:
        angle = -(90.0 + angle)
    elif angle > 45.0:
        angle = 90.0 - angle
    else:
        angle = -angle

    if abs(angle) > max_angle:
        return 0.0

    return float(angle)


def deskew_image(
    image: np.ndarray,
    binary_reference: Optional[np.ndarray] = None,
    max_angle: float = 45.0,
) -> Tuple[np.ndarray, float]:
    """Deskew image by rotating against detected skew angle."""
    if binary_reference is None:
        gray = to_grayscale(image)
        binary_reference = adaptive_threshold(gray, method="otsu")

    angle = compute_skew_angle(binary_reference, max_angle=max_angle)
    if abs(angle) < 0.2:
        return image.copy(), 0.0

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = np.abs(m[0, 0])
    sin = np.abs(m[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    m[0, 2] += (new_w / 2) - center[0]
    m[1, 2] += (new_h / 2) - center[1]

    border_val = (255, 255, 255) if image.ndim == 3 else 255
    deskewed = cv2.warpAffine(
        image,
        m,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_val,
    )
    return deskewed, angle


def filter_noise_boxes(
    boxes: List[Tuple[int, int, int, int, float, str]],
    min_width: int = 20,
    min_height: int = 10,
    min_area: int = 200,
    min_conf: float = 0.20,
) -> List[Tuple[int, int, int, int, float, str]]:
    """Remove small noise boxes, low-confidence detections, and isolated junk characters.

    Each box is (x1, y1, x2, y2, confidence, text).
    """
    valid = []
    junk_patterns = {"#", "# 0", "0", "~", "_", ".", "*", "-", "|"}

    for x1, y1, x2, y2, conf, text in boxes:
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        area = w * h
        clean_text = text.strip()

        # Reject if dimensions are smaller than minimal threshold
        if w < min_width or h < min_height or area < min_area:
            continue

        # Reject low confidence
        if conf < min_conf:
            continue

        # Reject isolated single-character junk symbols
        if len(clean_text) <= 2 and clean_text in junk_patterns and conf < 0.7:
            continue

        valid.append((x1, y1, x2, y2, conf, clean_text))

    return valid


def merge_word_boxes_into_lines(
    boxes_with_info: List[Tuple[int, int, int, int, float, str]],
    image_shape: Tuple[int, int],
    vertical_overlap_ratio: float = 0.35,
    padding_px: int = 6,
) -> List[Tuple[int, int, int, int, float, List[str]]]:
    """Merge detected word/phrase boxes into full horizontal text lines.

    TrOCR is designed for full line images. Feeding partial word fragments
    leads to token hallucinations.

    Returns:
        List of merged line tuples: (x1, y1, x2, y2, avg_conf, [words_in_line])
        sorted strictly top-to-bottom.
    """
    if not boxes_with_info:
        return []

    # Sort boxes primarily by vertical position (y1)
    sorted_boxes = sorted(boxes_with_info, key=lambda b: (b[1], b[0]))
    img_h, img_w = image_shape[:2]

    lines: List[List[Tuple[int, int, int, int, float, str]]] = []

    for box in sorted_boxes:
        bx1, by1, bx2, by2, bconf, btext = box
        bh = max(1, by2 - by1)
        b_mid_y = (by1 + by2) / 2.0

        matched_line_idx = None
        for idx, line in enumerate(lines):
            # Compute vertical extent and center of the line group
            line_y1 = min(item[1] for item in line)
            line_y2 = max(item[3] for item in line)
            line_h = max(1, line_y2 - line_y1)
            line_mid_y = (line_y1 + line_y2) / 2.0

            # Vertical overlap
            overlap_y1 = max(by1, line_y1)
            overlap_y2 = min(by2, line_y2)
            overlap_h = max(0, overlap_y2 - overlap_y1)
            ratio = overlap_h / min(bh, line_h)

            # Check overlap or vertical proximity
            if ratio >= vertical_overlap_ratio or abs(b_mid_y - line_mid_y) <= (0.45 * min(bh, line_h)):
                matched_line_idx = idx
                break

        if matched_line_idx is not None:
            lines[matched_line_idx].append(box)
        else:
            lines.append([box])

    merged_lines = []
    for line in lines:
        # Sort words in line from left to right
        line_sorted = sorted(line, key=lambda b: b[0])
        lx1 = min(b[0] for b in line_sorted)
        ly1 = min(b[1] for b in line_sorted)
        lx2 = max(b[2] for b in line_sorted)
        ly2 = max(b[3] for b in line_sorted)

        # Apply padding and clamp to image dimensions
        px1 = max(0, lx1 - padding_px)
        py1 = max(0, ly1 - padding_px)
        px2 = min(img_w, lx2 + padding_px)
        py2 = min(img_h, ly2 + padding_px)

        avg_conf = float(np.mean([b[4] for b in line_sorted]))
        words = [b[5] for b in line_sorted if b[5]]

        merged_lines.append((px1, py1, px2, py2, round(avg_conf, 4), words))

    # Sort merged lines top-to-bottom
    merged_lines.sort(key=lambda line: line[1])
    return merged_lines


def segment_lines(
    image: np.ndarray,
    min_line_height: int = 16,
    dilation_kernel_width: int = 25,
) -> List[np.ndarray]:
    """Segment an image into horizontal line crops using projection profiling."""
    gray = to_grayscale(image)
    binary = adaptive_threshold(gray, method="adaptive")
    inv = cv2.bitwise_not(binary)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation_kernel_width, 1))
    dilated = cv2.dilate(inv, kernel, iterations=2)

    proj = np.sum(dilated, axis=1)
    threshold = np.max(proj) * 0.05 if np.max(proj) > 0 else 0

    lines: List[np.ndarray] = []
    in_line = False
    start_y = 0

    for y, val in enumerate(proj):
        if val > threshold and not in_line:
            in_line = True
            start_y = y
        elif val <= threshold and in_line:
            in_line = False
            end_y = y
            if (end_y - start_y) >= min_line_height:
                pad = 4
                y1 = max(0, start_y - pad)
                y2 = min(image.shape[0], end_y + pad)
                line_crop = image[y1:y2, :]
                if line_crop.size > 0:
                    lines.append(line_crop)

    if in_line and (len(proj) - start_y) >= min_line_height:
        y1 = max(0, start_y - 4)
        lines.append(image[y1:, :])

    if not lines:
        lines = [image.copy()]

    return lines


def restore_and_enhance_pixels(
    image_input: np.ndarray,
    upscale_factor: float = 1.5,
    sharpen_strength: float = 0.5,
    contrast_clip: float = 2.0,
) -> Tuple[np.ndarray, dict]:
    """Restore degraded manuscript pixels, sharpen faded ink strokes, and upscale low-res folios.

    Features:
    1. Low-resolution detection & Super-Resolution upscaling via Lanczos-4 interpolation.
    2. Blur measurement (Laplacian variance) and Unsharp Masking for faded ink edge restoration.
    3. CLAHE luminance contrast enhancement to recover faded pigments without background burning.
    4. Bilateral edge-preserving smoothing to eliminate pixel noise and compression artifacts.
    """
    img = load_image(image_input)
    h, w = img.shape[:2]
    meta = {
        "original_resolution": (w, h),
        "upscaled": False,
        "sharpened": False,
        "contrast_enhanced": False,
        "initial_blur_variance": 0.0,
    }

    working_img = img.copy()
    # Only upscale genuinely low-resolution manuscripts (under 600px)
    if min(h, w) < 550 or max(h, w) < 750:
        target_w = int(w * upscale_factor)
        target_h = int(h * upscale_factor)
        working_img = cv2.resize(working_img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        meta["upscaled"] = True
        meta["new_resolution"] = (target_w, target_h)
    elif max(h, w) > 1600:
        # Cap oversized multi-megapixel images to 1500px to maintain fast CPU inference speed
        scale = 1500.0 / max(h, w)
        working_img = cv2.resize(working_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        meta["downscaled_for_speed"] = True

    gray = to_grayscale(working_img)
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    meta["initial_blur_variance"] = round(blur_var, 2)

    if blur_var < 350.0 or sharpen_strength > 0:
        gaussian = cv2.GaussianBlur(working_img, (0, 0), sigmaX=2.0)
        sharpened = cv2.addWeighted(working_img, 1.0 + sharpen_strength, gaussian, -sharpen_strength, 0)
        working_img = np.clip(sharpened, 0, 255).astype(np.uint8)
        meta["sharpened"] = True

    if working_img.ndim == 3 and working_img.shape[2] == 3:
        lab = cv2.cvtColor(working_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=contrast_clip, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        enhanced_lab = cv2.merge((l_enhanced, a, b))
        working_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        meta["contrast_enhanced"] = True

    working_img = cv2.bilateralFilter(working_img, d=5, sigmaColor=25, sigmaSpace=25)
    return working_img, meta


def preprocess_pipeline(
    image_input: np.ndarray,
    deskew: bool = True,
    denoise: bool = True,
    binarize: bool = False,
    enhance_pixels: bool = True,
    denoise_strength: int = 10,
    adaptive_block_size: int = 15,
    adaptive_c: int = 4,
) -> Tuple[np.ndarray, dict]:
    """Run configurable preprocessing chain on manuscript image."""
    img = load_image(image_input)
    meta = {
        "deskew_applied": False,
        "skew_angle": 0.0,
        "denoise_applied": False,
        "binarize_applied": False,
        "pixel_enhancement_applied": False,
    }

    if enhance_pixels:
        img, enh_meta = restore_and_enhance_pixels(img)
        meta["pixel_enhancement_applied"] = True
        meta["pixel_enhancement_details"] = enh_meta

    if deskew:
        img, angle = deskew_image(img)
        meta["deskew_applied"] = abs(angle) >= 0.2
        meta["skew_angle"] = round(angle, 2)

    gray = to_grayscale(img)

    if denoise:
        # Use fast bilateral filtering for edge-preserving parchment smoothing in milliseconds
        gray = denoise_image(gray, method="bilateral", strength=denoise_strength)
        meta["denoise_applied"] = True

    if binarize:
        processed = adaptive_threshold(
            gray, method="adaptive", block_size=adaptive_block_size, c=adaptive_c
        )
        meta["binarize_applied"] = True
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    else:
        processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    return processed, meta


@dataclass
class SegmentedRegion:
    """Represents a segmented line or word region with bounding box and crop."""
    bbox_xyxy: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    crop: np.ndarray
    index: int = 0
    words: List["SegmentedRegion"] = field(default_factory=list)


def segment_lines_morphological(
    image: np.ndarray,
    min_line_height: int = 10,
    min_line_width: int = 30,
    horizontal_kernel_width: int = 35,
) -> List[SegmentedRegion]:
    """Segment handwritten text lines using OpenCV morphological operations and contours."""
    gray = to_grayscale(image)
    _, bin_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_width, 3))
    dilated = cv2.dilate(bin_inv, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    h_img, w_img = image.shape[:2]
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h >= min_line_height and w >= min_line_width:
            x1 = max(0, x - 4)
            y1 = max(0, y - 2)
            x2 = min(w_img, x + w + 4)
            y2 = min(h_img, y + h + 2)
            boxes.append((x1, y1, x2, y2))

    boxes.sort(key=lambda b: (b[1], b[0]))
    regions = []
    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        crop = image[y1:y2, x1:x2].copy()
        regions.append(SegmentedRegion(bbox_xyxy=(x1, y1, x2, y2), crop=crop, index=idx))
    return regions


def segment_words_morphological(
    line_image: np.ndarray,
    min_word_width: int = 8,
    min_word_height: int = 8,
    horizontal_kernel_width: int = 7,
) -> List[SegmentedRegion]:
    """Segment individual words within a single text line crop using OpenCV morphology."""
    gray = to_grayscale(line_image)
    _, bin_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_width, 2))
    dilated = cv2.dilate(bin_inv, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_line, w_line = line_image.shape[:2]
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w >= min_word_width and h >= min_word_height:
            x1 = max(0, x - 2)
            y1 = max(0, y - 1)
            x2 = min(w_line, x + w + 2)
            y2 = min(h_line, y + h + 1)
            boxes.append((x1, y1, x2, y2))

    boxes.sort(key=lambda b: b[0])
    words = []
    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        crop = line_image[y1:y2, x1:x2].copy()
        words.append(SegmentedRegion(bbox_xyxy=(x1, y1, x2, y2), crop=crop, index=idx))
    return words


@dataclass
class AblationConfig:
    """Configuration to toggle each preprocessing stage independently for ablation studies."""
    grayscale: bool = True
    denoise: bool = True
    denoise_method: str = "bilateral"  # 'bilateral', 'fastNlMeans', 'gaussian'
    denoise_strength: int = 8
    deskew: bool = True
    deskew_max_angle: float = 45.0
    enhance_contrast: bool = False
    binarize: bool = False
    binarize_method: str = "sauvola"  # 'sauvola', 'otsu', 'adaptive'
    sauvola_window: int = 25
    sauvola_k: float = 0.2
    segment_lines: bool = False
    segment_words: bool = False


@dataclass
class AblationResult:
    """Detailed output and stage telemetry of an ablation preprocessing run."""
    output_image: np.ndarray
    stages: Dict[str, np.ndarray] = field(default_factory=dict)
    timings_ms: Dict[str, float] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    lines: List[SegmentedRegion] = field(default_factory=list)


def run_ablation_pipeline(
    image_input: Any,
    config: Optional[AblationConfig] = None,
    **kwargs: Any,
) -> AblationResult:
    """Execute fully modular preprocessing pipeline with each step independently toggleable.

    Supports systematic ablation experiments across:
    - Grayscale conversion
    - Denoising (bilateral, fastNlMeans, gaussian)
    - Deskewing (Hough / minAreaRect)
    - Contrast enhancement (CLAHE)
    - Adaptive binarization (Sauvola, Otsu, Gaussian adaptive)
    - Morphological line and word segmentation
    """
    if config is not None:
        cfg = config
    else:
        valid_fields = {f for f in AblationConfig.__dataclass_fields__}
        cfg_kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}
        cfg = AblationConfig(**cfg_kwargs)

    img = load_image(image_input)

    stages: Dict[str, np.ndarray] = {"original": img.copy()}
    timings: Dict[str, float] = {}
    metrics: Dict[str, Any] = {}
    working = img.copy()

    # 1. Grayscale
    if cfg.grayscale:
        t0 = time.perf_counter()
        gray = to_grayscale(working)
        timings["grayscale"] = round((time.perf_counter() - t0) * 1000, 2)
        stages["grayscale"] = gray.copy()
    else:
        gray = to_grayscale(working)

    # 2. Denoise
    if cfg.denoise:
        t0 = time.perf_counter()
        gray = denoise_image(gray, method=cfg.denoise_method, strength=cfg.denoise_strength)
        timings["denoise"] = round((time.perf_counter() - t0) * 1000, 2)
        stages["denoised"] = gray.copy()

    # 3. Deskew
    if cfg.deskew:
        t0 = time.perf_counter()
        gray, angle = deskew_image(gray, max_angle=cfg.deskew_max_angle)
        timings["deskew"] = round((time.perf_counter() - t0) * 1000, 2)
        metrics["skew_angle_deg"] = round(angle, 2)
        stages["deskewed"] = gray.copy()

    # 4. Contrast Enhancement
    if cfg.enhance_contrast:
        t0 = time.perf_counter()
        gray = enhance_contrast_mild(gray)
        timings["contrast_enhancement"] = round((time.perf_counter() - t0) * 1000, 2)
        stages["contrast_enhanced"] = gray.copy()

    # 5. Binarization
    if cfg.binarize:
        t0 = time.perf_counter()
        binary = adaptive_threshold(
            gray,
            method=cfg.binarize_method,
            block_size=cfg.sauvola_window,
            sauvola_k=cfg.sauvola_k,
        )
        timings["binarize"] = round((time.perf_counter() - t0) * 1000, 2)
        stages["binarized"] = binary.copy()
        working = binary
    else:
        working = gray

    if working.ndim == 2:
        output_display = cv2.cvtColor(working, cv2.COLOR_GRAY2BGR)
    else:
        output_display = working.copy()

    # 6. Line & Word Segmentation
    lines: List[SegmentedRegion] = []
    if cfg.segment_lines:
        t0 = time.perf_counter()
        lines = segment_lines_morphological(working)
        if cfg.segment_words:
            for line in lines:
                line.words = segment_words_morphological(line.crop)
        timings["segmentation"] = round((time.perf_counter() - t0) * 1000, 2)
        metrics["line_count"] = len(lines)
        metrics["total_words"] = sum(len(line.words) for line in lines)

    return AblationResult(
        output_image=output_display,
        stages=stages,
        timings_ms=timings,
        metrics=metrics,
        lines=lines,
    )
