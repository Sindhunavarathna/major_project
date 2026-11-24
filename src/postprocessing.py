"""Postprocessing module for Visual Document Translation.

Replaces original manuscript text regions with translated English text directly
onto the manuscript canvas, using background color sampling, seamless inpainting,
and dynamic typography fitting.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.preprocessing import TextLineRegion
from src.utils import bgr_to_pil, get_logger, load_image

LOGGER = get_logger("postprocessing")

# Font candidates for English manuscript rendering
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\cambria.ttc",
    r"C:\Windows\Fonts\times.ttf",
    r"C:\Windows\Fonts\georgia.ttf",
    r"C:\Windows\Fonts\ARIALUNI.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def get_available_font(preferred_path: Optional[str] = None) -> Optional[str]:
    """Find the best available serif/standard font for English manuscript text."""
    if preferred_path and Path(preferred_path).exists():
        return preferred_path
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return p
    return None


def sample_background_color(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    border_width: int = 3,
) -> Tuple[int, int, int]:
    """Sample the median background color along the border perimeter of a bounding box."""
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]

    # Collect border pixel coordinates
    border_pixels = []

    # Top and bottom strips
    y_top_start = max(0, y1 - border_width)
    y_top_end = max(0, y1)
    if y_top_end > y_top_start:
        border_pixels.append(image[y_top_start:y_top_end, max(0, x1):min(w, x2)])

    y_bot_start = min(h, y2)
    y_bot_end = min(h, y2 + border_width)
    if y_bot_end > y_bot_start:
        border_pixels.append(image[y_bot_start:y_bot_end, max(0, x1):min(w, x2)])

    # Left and right strips
    x_left_start = max(0, x1 - border_width)
    x_left_end = max(0, x1)
    if x_left_end > x_left_start:
        border_pixels.append(image[max(0, y1):min(h, y2), x_left_start:x_left_end])

    x_right_start = min(w, x2)
    x_right_end = min(w, x2 + border_width)
    if x_right_end > x_right_start:
        border_pixels.append(image[max(0, y1):min(h, y2), x_right_start:x_right_end])

    valid_strips = [s.reshape(-1, 3) for s in border_pixels if s.size > 0]
    if valid_strips:
        all_pixels = np.vstack(valid_strips)
        median_color = np.median(all_pixels, axis=0).astype(int)
        return int(median_color[0]), int(median_color[1]), int(median_color[2])

    # Fallback to general parchment warm tint
    return (230, 244, 248)


def inpaint_text_region(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    use_telea: bool = True,
) -> np.ndarray:
    """Seamlessly remove original text inside the bounding box while preserving paper grain."""
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]

    # Bound coordinates
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return image

    crop = image[y1:y2, x1:x2]
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop

    # Identify dark ink text pixels to inpaint
    # Using Otsu thresholding or adaptive thresholding to create text mask
    _, binary_inv = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilate ink mask slightly to cover antialiased edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(binary_inv, kernel, iterations=1)

    if np.sum(mask > 0) > 0 and use_telea:
        try:
            # OpenCV inpainting on the crop
            inpainted_crop = cv2.inpaint(crop, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            image[y1:y2, x1:x2] = inpainted_crop
            return image
        except Exception as e:
            LOGGER.debug("cv2.inpaint failed, falling back to median color fill: %s", e)

    # Fallback: fill with sampled parchment background color with feathered boundary
    bg_color = sample_background_color(image, (x1, y1, x2, y2))
    filled_crop = np.full_like(crop, bg_color, dtype=np.uint8)

    # Slight Gaussian feathering around borders to eliminate seams
    feather_mask = np.ones((y2 - y1, x2 - x1), dtype=np.float32)
    border_px = 3
    if border_px > 0 and (y2 - y1) > 2 * border_px and (x2 - x1) > 2 * border_px:
        cv2.rectangle(feather_mask, (0, 0), (x2 - x1 - 1, y2 - y1 - 1), 0.0, border_px)
        feather_mask = cv2.GaussianBlur(feather_mask, (5, 5), 0)
        feather_mask = np.dstack([feather_mask] * 3)
        blended = (crop * (1.0 - feather_mask) + filled_crop * feather_mask).astype(np.uint8)
        image[y1:y2, x1:x2] = blended
    else:
        image[y1:y2, x1:x2] = filled_crop

    return image


def fit_text_lines_to_box(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_w: int,
    max_h: int,
    font_path: Optional[str] = None,
    min_font_size: int = 10,
    max_font_size: int = 32,
) -> Tuple[Optional[Any], List[str]]:
    """Determine optimal font size and word wrapping to fit text cleanly into a bounding box."""
    words = text.strip().split()
    if not words:
        return None, []

    font_file = get_available_font(font_path)

    for font_size in range(max_font_size, min_font_size - 1, -2):
        try:
            if font_file and font_file.endswith(".ttc"):
                font = ImageFont.truetype(font_file, font_size, index=0)
            elif font_file:
                font = ImageFont.truetype(font_file, font_size)
            else:
                font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

        # Word wrap test
        lines: List[str] = []
        current_line = ""

        fits = True
        for word in words:
            test_line = f"{current_line} {word}".strip()
            # Get text width
            bbox = draw.textbbox((0, 0), test_line, font=font)
            line_w = bbox[2] - bbox[0]

            if line_w <= max_w:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                    current_line = word
                    # Check if single word exceeds width
                    w_bbox = draw.textbbox((0, 0), word, font=font)
                    if (w_bbox[2] - w_bbox[0]) > max_w:
                        fits = False
                        break
                else:
                    fits = False
                    break

        if current_line:
            lines.append(current_line)

        if not fits:
            continue

        # Check total height of lines
        line_height = font_size + 4
        total_h = len(lines) * line_height

        if total_h <= max_h and lines:
            return font, lines

    # Smallest fallback
    try:
        font = (
            ImageFont.truetype(font_file, min_font_size)
            if font_file
            else ImageFont.load_default()
        )
    except Exception:
        font = ImageFont.load_default()

    return font, [text]


def create_translated_image(
    image: Union[np.ndarray, Image.Image, str, Path],
    line_regions: List[Union[TextLineRegion, Tuple[int, int, int, int]]],
    translated_text: str,
    font_color: Tuple[int, int, int] = (25, 20, 15),  # Classic dark manuscript ink
    use_inpainting: bool = True,
    font_path: Optional[str] = None,
) -> np.ndarray:
    """Generate a visual document translation where foreign text is replaced by English text.

    Args:
        image: Original manuscript image (numpy array, PIL image, or path).
        line_regions: Detected line regions (TextLineRegion objects or bounding boxes (x1, y1, x2, y2)).
        translated_text: Full translated English text (multi-line or paragraph).
        font_color: RGB text color (default dark ink brown/black).
        use_inpainting: Whether to use texture inpainting on text ink strokes.
        font_path: Optional custom TTF/OTF font path.

    Returns:
        np.ndarray: BGR image with the translated English text rendered onto original page.
    """
    img_bgr = load_image(image).copy()
    h, w = img_bgr.shape[:2]

    # Normalize boxes
    boxes: List[Tuple[int, int, int, int]] = []
    if line_regions:
        for r in line_regions:
            if isinstance(r, TextLineRegion):
                boxes.append(r.bbox_xyxy)
            elif isinstance(r, (list, tuple)) and len(r) == 4:
                boxes.append((int(r[0]), int(r[1]), int(r[2]), int(r[3])))

    if not boxes or not translated_text.strip():
        # Overlay clear status notification so user is never misled by a raw un-translated image
        canvas = img_bgr.copy()
        overlay = canvas.copy()
        msg = "Could not recognize text in this image"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        thick = 2
        text_size = cv2.getTextSize(msg, font, scale, thick)[0]
        text_x = max(10, (w - text_size[0]) // 2)
        text_y = max(30, (h + text_size[1]) // 2)
        cv2.rectangle(
            overlay,
            (text_x - 12, text_y - text_size[1] - 12),
            (text_x + text_size[0] + 12, text_y + 12),
            (245, 245, 245),
            -1,
        )
        cv2.rectangle(
            overlay,
            (text_x - 12, text_y - text_size[1] - 12),
            (text_x + text_size[0] + 12, text_y + 12),
            (60, 60, 220),
            2,
        )
        cv2.putText(overlay, msg, (text_x, text_y), font, scale, (40, 40, 200), thick, cv2.LINE_AA)
        return cv2.addWeighted(overlay, 0.90, canvas, 0.10, 0)

    # Sort boxes top-to-bottom
    boxes.sort(key=lambda b: b[1])

    # Inpaint / clean the original text from all regions first
    for bbox in boxes:
        img_bgr = inpaint_text_region(img_bgr, bbox, use_telea=use_inpainting)

    # Convert to PIL RGB for clean font typography
    pil_img = bgr_to_pil(img_bgr)
    draw = ImageDraw.Draw(pil_img)

    # Split translated text into lines
    raw_trans_lines = [l.strip() for l in translated_text.split("\n") if l.strip()]
    if not raw_trans_lines:
        raw_trans_lines = [translated_text.strip()]

    # Map translated text chunks onto available bounding boxes
    num_boxes = len(boxes)
    num_lines = len(raw_trans_lines)

    if num_lines <= num_boxes:
        assigned_texts = raw_trans_lines + [""] * (num_boxes - num_lines)
    else:
        # If more translated lines than boxes, pack text proportionally
        assigned_texts = []
        words = translated_text.split()
        words_per_box = max(1, len(words) // num_boxes)
        for i in range(num_boxes):
            if i == num_boxes - 1:
                chunk = " ".join(words[i * words_per_box:])
            else:
                chunk = " ".join(words[i * words_per_box:(i + 1) * words_per_box])
            assigned_texts.append(chunk)

    # Render English text onto each cleaned box
    for bbox, text_chunk in zip(boxes, assigned_texts):
        if not text_chunk:
            continue

        x1, y1, x2, y2 = bbox
        box_w = max(20, x2 - x1)
        box_h = max(15, y2 - y1)

        # Pad slightly inside the box
        margin_x = 4
        margin_y = 2
        avail_w = max(10, box_w - (2 * margin_x))
        avail_h = max(10, box_h - (2 * margin_y))

        font, wrapped_lines = fit_text_lines_to_box(
            draw=draw,
            text=text_chunk,
            max_w=avail_w,
            max_h=avail_h,
            font_path=font_path,
        )

        if not wrapped_lines:
            continue

        # Calculate line spacing and vertical centering
        line_spacing = 3
        sample_bbox = draw.textbbox((0, 0), "Ay", font=font)
        single_h = sample_bbox[3] - sample_bbox[1]
        total_text_h = len(wrapped_lines) * (single_h + line_spacing)

        start_y = y1 + margin_y + max(0, (box_h - total_text_h) // 2)

        curr_y = start_y
        for w_line in wrapped_lines:
            line_bbox = draw.textbbox((0, 0), w_line, font=font)
            curr_line_w = line_bbox[2] - line_bbox[0]
            # Center or left align based on box width
            start_x = x1 + margin_x
            draw.text((start_x, curr_y), w_line, fill=font_color, font=font)
            curr_y += single_h + line_spacing

    # Convert back to OpenCV BGR
    result_rgb = np.array(pil_img)
    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    return result_bgr

