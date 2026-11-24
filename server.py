"""
FastAPI Server for Multilingual Handwritten Note Recognition & Translation.

Provides:
- GET /: Serves responsive single-page HTML/CSS/JS web dashboard.
- GET /api/status: Health check and loaded model telemetry.
- GET /api/samples: Pre-configured sample manuscripts.
- POST /predict: High-performance inference endpoint returning JSON with:
    - detected language, native name, script, confidence, and uncertainty flags.
    - recognized text (with line-by-line bounding boxes).
    - English translation (NLLB-200 distilled quantized).
    - high-resolution stage-by-stage timings (preprocess, script_classifier, ocr, language_id, translation).
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.fast_pipeline import FastPipelineManager, FastPipelineResult
from src.utils import get_logger, load_image

LOGGER = get_logger("fastapi_server")

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
FRONTEND_DIR.mkdir(exist_ok=True)
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="Multilingual Handwritten Note Recognition & Translation API",
    description="High-performance paleography and offline translation API covering Kannada, Tamil, Telugu, German, and Arabic.",
    version="2.0.0",
)

# Enable CORS for local and web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount sample_data for frontend preview thumbnails
SAMPLE_DATA_DIR = BASE_DIR / "sample_data"
SAMPLE_DATA_DIR.mkdir(exist_ok=True)
app.mount("/sample_data", StaticFiles(directory=str(SAMPLE_DATA_DIR)), name="sample_data")

# Pre-defined sample manuscripts for one-click testing
SAMPLE_MANUSCRIPTS = [
    {
        "id": "kannada",
        "title": "Kannada Palm-Leaf Note",
        "language": "Kannada",
        "script": "Kannada",
        "path": "sample_data/sample_kannada.png",
    },
    {
        "id": "tamil",
        "title": "Tamil Epigraphic Script",
        "language": "Tamil",
        "script": "Tamil",
        "path": "sample_data/sample_tamil.png",
    },
    {
        "id": "telugu",
        "title": "Telugu Classical Manuscript",
        "language": "Telugu",
        "script": "Telugu",
        "path": "sample_data/sample_telugu.png",
    },
    {
        "id": "german",
        "title": "German Kurrent Diary",
        "language": "German",
        "script": "Latin",
        "path": "sample_data/sample_german.png",
    },
    {
        "id": "arabic",
        "title": "Classical Arabic Folio",
        "language": "Arabic",
        "script": "Arabic",
        "path": "sample_data/sample_arabic.png",
    },
    {
        "id": "devanagari",
        "title": "Classical Sanskrit Manuscript (Rigveda 2.27)",
        "language": "Sanskrit",
        "script": "Devanagari",
        "path": "sample_data/sample_sanskrit.png",
    },
    {
        "id": "sanskrit",
        "title": "Classical Sanskrit Manuscript (Rigveda 2.27)",
        "language": "Sanskrit",
        "script": "Devanagari",
        "path": "sample_data/sample_sanskrit.png",
    },
]


@app.on_event("startup")
async def startup_event():
    """Pre-load all models into memory at server startup."""
    LOGGER.info("Starting up FastAPI application. Pre-loading models...")
    _ = FastPipelineManager.get_instance()
    LOGGER.info("FastAPI ready to receive inference requests.")


@app.get("/")
async def root():
    """Serve the single-page HTML frontend."""
    frontend_file = FRONTEND_DIR / "index.html"
    if frontend_file.exists():
        return FileResponse(frontend_file)
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse({
        "message": "Multilingual Handwriting Translation API is running. frontend/index.html not found.",
        "docs": "/docs",
    })


@app.get("/api/status")
async def get_status():
    """Return health check and model telemetry."""
    manager = FastPipelineManager.get_instance()
    return {
        "status": "online",
        "models": {
            "script_classifier": manager.script_classifier is not None,
            "indic_crnn": manager.indic_crnn is not None and manager.indic_crnn.is_available,
            "easyocr": manager.easyocr is not None,
            "trocr": manager.trocr is not None,
            "translator": manager.translator is not None,
        },
        "quantization": "int8_dynamic_cpu",
        "supported_languages": ["Kannada", "Tamil", "Telugu", "Arabic", "German", "English"],
    }


@app.get("/api/samples")
async def get_samples():
    """Return list of sample manuscripts available on disk."""
    return {"samples": SAMPLE_MANUSCRIPTS}


@app.post("/predict")
async def predict_endpoint(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
    sample_id: Optional[str] = Form(None),
    model_mode: str = Form("custom"),  # "custom" or "baseline"
    script_hint: Optional[str] = Form("auto"),
    input_type: str = Form("auto"),  # "auto", "handwritten", "printed"
    enable_cloud_bridge: bool = Form(False),  # Explicit, visibly-labelled opt-in cloud translation
    confidence_threshold: float = Form(0.35),  # Configurable confidence gate threshold before translation
    deskew: bool = Form(True),
    denoise: bool = Form(True),
    binarize: bool = Form(True),
):
    """Execute end-to-end recognition & translation.

    Accepts an uploaded file, a base64 encoded image (from camera/clipboard), or a sample_id.
    """
    manager = FastPipelineManager.get_instance()
    image_data: Optional[Any] = None

    # 1. Check for sample_id
    if sample_id:
        matched = next((s for s in SAMPLE_MANUSCRIPTS if s["id"] == sample_id), None)
        if matched and Path(matched["path"]).exists():
            image_data = matched["path"]
        else:
            raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found.")

    # 2. Check for uploaded file
    elif file is not None:
        try:
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            image_data = content
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {e}")

    # 3. Check for image_base64
    elif image_base64:
        try:
            # Strip data URL header if present (e.g. 'data:image/png;base64,...')
            if "," in image_base64:
                image_base64 = image_base64.split(",", 1)[1]
            image_bytes = base64.b64decode(image_base64)
            image_data = image_bytes
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {e}")

    else:
        raise HTTPException(
            status_code=400,
            detail="Must provide an image via 'file' upload, 'image_base64', or 'sample_id'.",
        )

    # Execute inference
    try:
        result = manager.process(
            image_input=image_data,
            model_mode=model_mode,
            script_hint=script_hint,
            input_type=input_type,
            enable_cloud_bridge=enable_cloud_bridge,
            confidence_threshold=confidence_threshold,
            deskew=deskew,
            denoise=denoise,
            binarize=binarize,
        )
        return JSONResponse(result.to_dict())
    except Exception as exc:
        LOGGER.exception("Inference failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")


@app.post("/api/download")
async def download_translation_report(
    title: str = Form("Manuscript Translation"),
    detected_language: str = Form("Unknown"),
    detected_script: str = Form("Unknown"),
    confidence: float = Form(0.0),
    recognized_text: str = Form(""),
    translated_text: str = Form(""),
    format: str = Form("pdf"),
):
    """Generate and return downloadable translation report in PDF or Text format."""
    import io
    import time
    from fastapi.responses import Response

    clean_fmt = format.lower().strip()
    if clean_fmt == "pdf":
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontSize=16,
            leading=20,
            textColor=colors.HexColor("#1E293B"),
            spaceAfter=8,
        )
        meta_label_style = ParagraphStyle(
            "MetaLabel",
            parent=styles["Normal"],
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#475569"),
        )
        section_style = ParagraphStyle(
            "SectionTitle",
            parent=styles["Heading2"],
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#4F46E5"),
            spaceBefore=10,
            spaceAfter=6,
        )
        body_style = ParagraphStyle(
            "Body",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#0F172A"),
        )

        story = []
        story.append(Paragraph(f"<b>PolyScribe AI — {title}</b>", title_style))
        story.append(Paragraph("Multilingual Handwritten Note Recognition + Translation Report", meta_label_style))
        story.append(Spacer(1, 10))

        table_data = [
            [
                Paragraph("<b>Detected Language:</b>", meta_label_style),
                Paragraph(detected_language, body_style),
                Paragraph("<b>Script:</b>", meta_label_style),
                Paragraph(detected_script, body_style),
            ],
            [
                Paragraph("<b>Confidence:</b>", meta_label_style),
                Paragraph(f"{confidence:.1%}", body_style),
                Paragraph("<b>Generated:</b>", meta_label_style),
                Paragraph(time.strftime("%Y-%m-%d %H:%M:%S"), body_style),
            ],
        ]
        meta_table = Table(table_data, colWidths=[110, 150, 110, 150])
        meta_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 12))

        story.append(Paragraph("1. Recognized Original Text", section_style))
        clean_rec = recognized_text.replace("\n", "<br/>") if recognized_text.strip() else "<i>(Could not recognize text in this image)</i>"
        story.append(Paragraph(clean_rec, body_style))
        story.append(Spacer(1, 12))

        story.append(Paragraph("2. English Translation", section_style))
        clean_trans = translated_text.replace("\n", "<br/>") if translated_text.strip() else "<i>(Translation unavailable)</i>"
        story.append(Paragraph(clean_trans, body_style))

        doc.build(story)
        buf.seek(0)
        return Response(
            content=buf.getvalue(),
            media_type="application/pdf",
            headers={"Content-Disposition": 'attachment; filename="translation_report.pdf"'},
        )
    else:
        text_content = (
            f"PolyScribe AI Translation Report\n"
            f"===============================\n"
            f"Detected Language: {detected_language}\n"
            f"Script: {detected_script}\n"
            f"Confidence: {confidence:.1%}\n\n"
            f"--- RECOGNIZED TEXT ---\n{recognized_text}\n\n"
            f"--- ENGLISH TRANSLATION ---\n{translated_text}\n"
        )
        return Response(
            content=text_content,
            media_type="text/plain",
            headers={"Content-Disposition": 'attachment; filename="translation_report.txt"'},
        )


# Mount static directory for assets if needed
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
