"""Dataset generation, augmentations, and PyTorch DataLoader for script classification.

Supports 5 classes: Kannada, Tamil, Telugu, Arabic, Latin.
Generates robust handwritten-style character/word patches with:
- Elastic-like affine shear & rotation (+/-15 deg)
- Variable stroke pen thickness (morphological dilation / erosion)
- Gaussian blur & parchment texture noise
- Strict train / val / test splitting without sample leakage.
"""

import os
import random
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from src.script_classifier.model import CLASS_NAMES, CLASS_TO_IDX

# Authentic corpus words and character n-grams per script
SCRIPT_TEXT_CORPUS: Dict[str, List[str]] = {
    "Kannada": [
        "ಕನ್ನಡ", "ನಮಸ್ಕಾರ", "ಪುಸ್ತಕ", "ಶಾಲೆ", "ಇತಿಹಾಸ", "ವಿದ್ಯಾರ್ಥಿ", "ಭಾರತ",
        "ಮನುಷ್ಯ", "ಧರ್ಮ", "ಶಾಂತಿ", "ಸಾಹಿತ್ಯ", "ಕಲೆ", "ಪ್ರೀತಿ", "ಜೀವನ", "ಜ್ಞಾನ",
        "ಕೃಪೆ", "ಬರಹ", "ಲಿಪಿ", "ಹಸ್ತಪ್ರತಿ", "ಸಂಸ್ಕೃತಿ", "ಭಾಷೆ", "ರಾಜ್ಯ", "ಜನರು",
        "ದೇಶ", "ಲೋಕ", "ದೇವ", "ಗುರು", "ಸತ್ಯ", "ಕಾವ್ಯ", "ಕವನ", "ಕಥೆ", "ಹಾಡು",
        "ಬೆಳಕು", "ಗಾಳಿ", "ನೀರು", "ಭೂಮಿ", "ಆಕಾಶ", "ಕಣ್ಣು", "ಮನಸ್ಸು", "ಹೃದಯ",
    ],
    "Tamil": [
        "தமிழ்", "வணக்கம்", "புத்தகம்", "பள்ளி", "வரலாறு", "மாணவன்", "பாரதம்",
        "மனிதன்", "அறம்", "அமைதி", "இலக்கியம்", "கலை", "அன்பு", "வாழ்க்கை", "அறிவு",
        "எழுத்து", "சுவடி", "பண்பாடு", "மொழி", "நாடு", "மக்கள்", "உலகம்", "இறைவன்",
        "ஆசான்", "உண்மை", "கவிதை", "பாடல்", "கதை", "ஒளி", "காற்று", "நீர்",
        "நிலம்", "வானம்", "கண்", "மனம்", "இதயம்", "வெற்றி", "நம்பிக்கை", "காலம்",
    ],
    "Telugu": [
        "తెలుగు", "నమస్కారం", "పుస్తకం", "పాఠశాల", "చరిత్ర", "విద్యార్థి", "భారతం",
        "మానవుడు", "ధర్మం", "శాంతి", "సాహిత్యం", "కళ", "ప్రేమ", "జీవితం", "జ్ఞానం",
        "రాత", "లిపి", "తాళపత్రం", "సంస్కృతి", "భాష", "రాష్ట్రం", "ప్రజలు", "దేశం",
        "లోకం", "దేవుడు", "గురువు", "సత్యం", "కావ్యం", "కవిత", "కథ", "పాట",
        "వెలుగు", "గాలి", "నీరు", "భూమి", "ఆకాశం", "కన్ను", "మనస్సు", "హృదయం",
    ],
    "Arabic": [
        "العربية", "سلام", "كتاب", "مدرسة", "تاريخ", "طالب", "إنسان", "عدل",
        "أدب", "فن", "حب", "حياة", "علم", "مخطوطة", "قراءة", "حكمة", "نور",
        "خط", "لغة", "وطن", "ناس", "عالم", "أستاذ", "حق", "شعر", "قصيدة",
        "قصة", "هواء", "ماء", "أرض", "سماء", "عين", "قلب", "نجاح", "أمل",
    ],
    "Latin": [
        "Monat", "Maerz", "Sonntag", "Morgen", "Handschrift", "Wissen", "Weisheit",
        "Wahrheit", "Jahrhunderte", "Darin", "Buch", "Leben", "Mensch", "Gott",
        "Deutschland", "manuscript", "history", "knowledge", "science", "reading",
        "library", "translation", "language", "document", "parchment", "ancient",
        "beautiful", "sunday", "morning", "writing", "cursive", "letter", "words",
    ],
    "Devanagari": [
        "संस्कृतम्", "नमस्ते", "पुस्तकम्", "विद्यालय", "इतिहास", "विद्यार्थी", "भारतम्",
        "मनुष्य", "धर्म", "शान्ति", "साहित्य", "कला", "प्रेम", "जीवन", "ज्ञानम्",
        "ऋग्वेद", "संहिता", "वरुण", "मित्र", "आदित्य", "ऋतम्", "सत्यम्", "मण्डल",
        "सूक्त", "श्लोक", "मन्त्र", "अग्नि", "इन्द्र", "सोम", "देवता", "प्रार्थना",
        "प्रकाश", "वायु", "जलम्", "भूमि", "आकाश", "नेत्र", "मनस्", "हृदयम्", "शान्तिः",
    ],
}


def _get_system_font(size: int = 32) -> ImageFont.ImageFont:
    """Attempt to locate a multi-script TTF font on Windows; fallback to default."""
    candidate_fonts = [
        r"C:\Windows\Fonts\Nirmala.ttf",
        r"C:\Windows\Fonts\Arial.ttf",
        r"C:\Windows\Fonts\SegUIReg.ttf",
        r"C:\Windows\Fonts\times.ttf",
        r"C:\Windows\Fonts\tahoma.ttf",
    ]
    for fp in candidate_fonts:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return ImageFont.load_default()


def render_script_patch(
    text: str,
    target_size: Tuple[int, int] = (128, 128),
    font_size: int = 34,
) -> np.ndarray:
    """Render a text sample onto a realistic parchment-like single-channel 128x128 patch."""
    font = _get_system_font(font_size)
    bg_color = random.randint(235, 255)
    ink_color = random.randint(15, 60)

    # Canvas with random margin
    canvas_w = max(180, target_size[0] + 60)
    canvas_h = max(180, target_size[1] + 60)
    img = Image.new("L", (canvas_w, canvas_h), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Draw centered text with subtle jitter
    ox = random.randint(20, 40)
    oy = random.randint(30, 60)
    draw.text((ox, oy), text, fill=ink_color, font=font)

    arr = np.array(img, dtype=np.uint8)

    # Center crop or resize to target_size
    h, w = arr.shape
    tw, th = target_size
    cy, cx = h // 2, w // 2
    y1 = max(0, cy - th // 2)
    y2 = min(h, y1 + th)
    x1 = max(0, cx - tw // 2)
    x2 = min(w, x1 + tw)
    patch = arr[y1:y2, x1:x2]
    if patch.shape != (th, tw):
        patch = cv2.resize(patch, (tw, th), interpolation=cv2.INTER_AREA)

    return patch


def apply_handwriting_augmentations(patch: np.ndarray) -> np.ndarray:
    """Apply realistic handwriting distortions and historical parchment noise."""
    h, w = patch.shape[:2]
    out = patch.copy()

    # 1. Random rotation (-14 to +14 deg) and slight shear
    angle = random.uniform(-14.0, 14.0)
    shear = random.uniform(-0.15, 0.15)
    scale = random.uniform(0.90, 1.10)
    center = (w / 2, h / 2)
    m = cv2.getRotationMatrix2D(center, angle, scale)
    m[0, 1] += shear
    out = cv2.warpAffine(out, m, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # 2. Variable pen stroke thickness (morphological dilation / erosion)
    k_choice = random.random()
    if k_choice < 0.25:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        out = cv2.erode(out, kernel, iterations=1)  # thicken dark ink
    elif k_choice < 0.45:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        out = cv2.dilate(out, kernel, iterations=1)  # thin ink strokes

    # 3. Additive Gaussian noise (parchment speckles)
    if random.random() < 0.70:
        noise = np.random.normal(0, random.uniform(3.0, 8.0), (h, w))
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 4. Subtle Gaussian blur
    if random.random() < 0.50:
        k = random.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), sigmaX=random.uniform(0.4, 0.9))

    return out


class ScriptDataset(Dataset):
    """PyTorch Dataset yielding (1, 128, 128) float tensor patches and integer class label."""

    def __init__(self, patches: List[np.ndarray], labels: List[int], augment: bool = False):
        self.patches = patches
        self.labels = labels
        self.augment = augment

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        patch = self.patches[idx]
        if self.augment:
            patch = apply_handwriting_augmentations(patch)

        # Normalize to [-1.0, 1.0]
        norm = (patch.astype(np.float32) / 127.5) - 1.0
        tensor = torch.from_numpy(norm).unsqueeze(0)  # Shape: (1, 128, 128)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return tensor, label


def generate_synthetic_script_data(
    samples_per_class: int = 150,
    seed: int = 42,
) -> Tuple[List[np.ndarray], List[int]]:
    """Synthesize diverse manuscript script patches across all 5 classes."""
    random.seed(seed)
    np.random.seed(seed)

    all_patches: List[np.ndarray] = []
    all_labels: List[int] = []

    for class_name, class_idx in CLASS_TO_IDX.items():
        corpus = SCRIPT_TEXT_CORPUS.get(class_name, ["test"])
        for i in range(samples_per_class):
            word = random.choice(corpus)
            # Sometimes render 2-word combinations
            if random.random() < 0.35 and len(corpus) > 1:
                word = f"{word} {random.choice(corpus)}"

            font_sz = random.randint(24, 38)
            patch = render_script_patch(word, target_size=(128, 128), font_size=font_sz)
            patch = apply_handwriting_augmentations(patch)
            all_patches.append(patch)
            all_labels.append(class_idx)

    return all_patches, all_labels


def get_train_val_test_dataloaders(
    batch_size: int = 32,
    samples_per_class: int = 150,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """Create train, validation, and test DataLoader objects with strict splits."""
    patches, labels = generate_synthetic_script_data(samples_per_class=samples_per_class, seed=seed)
    n_total = len(patches)

    indices = list(range(n_total))
    random.seed(seed)
    random.shuffle(indices)

    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    train_idx = indices[:n_train]
    val_idx = indices[n_train: n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_patches = [patches[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]

    val_patches = [patches[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    test_patches = [patches[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    train_ds = ScriptDataset(train_patches, train_labels, augment=True)
    val_ds = ScriptDataset(val_patches, val_labels, augment=False)
    test_ds = ScriptDataset(test_patches, test_labels, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    split_counts = {
        "total": n_total,
        "train": len(train_idx),
        "val": len(val_idx),
        "test": len(test_idx),
    }

    return train_loader, val_loader, test_loader, split_counts
