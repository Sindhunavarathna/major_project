"""Generate synthetic test images for OCR, LangID, and translation testing."""

import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


SAMPLE_DIR = Path(__file__).resolve().parent
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES = [
    {
        "filename": "sample_english.png",
        "lang": "en",
        "script": "Latin",
        "text": "Ancient manuscripts preserve the knowledge and philosophy of centuries.",
        "translation_en": "Ancient manuscripts preserve the knowledge and philosophy of centuries.",
    },
    {
        "filename": "sample_hindi.png",
        "lang": "hi",
        "script": "Devanagari",
        "text": "यह एक प्राचीन पांडुलिपि का नमूना है।",
        "translation_en": "This is a sample of an ancient manuscript.",
    },
    {
        "filename": "sample_sanskrit.png",
        "lang": "sa",
        "script": "Devanagari",
        "text": "सर्वे भवन्तु सुखिनः सर्वे सन्तु निरामयाः ।",
        "translation_en": "May all beings be happy, may all beings be free from illness.",
    },
    {
        "filename": "sample_kannada.png",
        "lang": "kn",
        "script": "Kannada",
        "text": "ಪ್ರಾಚೀನ ಹಸ್ತಪ್ರತಿಗಳು ಭಾರತದ ಇತಿಹಾಸವನ್ನು ತಿಳಿಸುತ್ತವೆ.",
        "translation_en": "Ancient manuscripts tell the history of India.",
    },
    {
        "filename": "sample_tamil.png",
        "lang": "ta",
        "script": "Tamil",
        "text": "பண்டைய கையெழுத்துப் பிரதிகள் இந்தியாவின் வரலாற்றைக் கூறுகின்றன.",
        "translation_en": "Ancient manuscripts tell the history of India.",
    },
    {
        "filename": "sample_telugu.png",
        "lang": "te",
        "script": "Telugu",
        "text": "పురాతన రాతప్రతులు భారతదేశ చరిత్రను తెలియజేస్తాయి.",
        "translation_en": "Ancient manuscripts convey the history of India.",
    },
]


def create_sample_images():
    font_candidates = [
        r"C:\Windows\Fonts\Nirmala.ttc",
        r"C:\Windows\Fonts\ARIALUNI.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    font_path = None
    for p in font_candidates:
        if Path(p).exists():
            font_path = p
            break

    for item in SAMPLES:
        # Create a document/manuscript parchment style background
        width, height = 800, 200
        # Warm parchment tint: #f8f4e6
        img = Image.new("RGB", (width, height), color=(248, 244, 230))
        draw = ImageDraw.Draw(img)

        # Draw subtle border
        draw.rectangle([(10, 10), (width - 10, height - 10)], outline=(180, 160, 130), width=2)

        try:
            if font_path and font_path.endswith(".ttc"):
                font = ImageFont.truetype(font_path, 28, index=0)
            elif font_path:
                font = ImageFont.truetype(font_path, 28)
            else:
                font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

        # Text position
        draw.text((30, 80), item["text"], fill=(30, 30, 30), font=font)

        out_path = SAMPLE_DIR / item["filename"]
        img.save(out_path)
        print(f"Created {out_path}")

    # Write samples.json metadata
    json_path = SAMPLE_DIR / "samples.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(SAMPLES, f, ensure_ascii=False, indent=2)
    print(f"Created {json_path}")


if __name__ == "__main__":
    create_sample_images()

