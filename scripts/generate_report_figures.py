"""
Script to generate publication/report-ready architecture and pipeline flow diagrams.
Saves:
  - docs/project_architecture_flow.png
  - docs/indic_crnn_architecture.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

os.makedirs("docs", exist_ok=True)

def generate_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.axis("off")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8.5)

    # Title
    ax.text(7, 8.1, "Multilingual Handwritten Note Recognition & Translation Architecture", 
            ha="center", va="center", fontsize=15, fontweight="bold", color="#1E293B")
    ax.text(7, 7.7, "Final-Year Major Project Pipeline: Kannada | Tamil | Telugu | German | Arabic", 
            ha="center", va="center", fontsize=11, fontstyle="italic", color="#475569")

    # Helper function for boxes
    def draw_box(x, y, w, h, title, subtitle, color, border_color="#334155"):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15", 
                                      linewidth=1.8, edgecolor=border_color, facecolor=color)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.65, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color="#0F172A")
        if subtitle:
            ax.text(x + w/2, y + h*0.28, subtitle, ha="center", va="center", fontsize=8.5, color="#334155")

    # 1. Input Image
    draw_box(0.5, 4.5, 2.2, 1.8, "1. Input Document", "Manuscript / Note Scan\n(5 Target Languages)", "#E0F2FE", "#0284C7")

    # Arrow 1 -> 2
    ax.annotate("", xy=(3.1, 5.4), xytext=(2.7, 5.4),
                arrowprops=dict(facecolor="#0284C7", edgecolor="#0284C7", width=2, headwidth=7))

    # 2. Preprocessing Module
    draw_box(3.1, 4.2, 2.6, 2.4, "2. Preprocessing\n(OpenCV)", "• Deskew (Hough lines)\n• FastNLMeans Denoising\n• Sauvola Binarization\n• Line & Word Slicing\n[Toggleable Ablations]", "#FEF3C7", "#D97706")

    # Arrow 2 -> 3
    ax.annotate("", xy=(6.1, 5.4), xytext=(5.7, 5.4),
                arrowprops=dict(facecolor="#D97706", edgecolor="#D97706", width=2, headwidth=7))

    # 3. Script Classifier
    draw_box(6.1, 4.4, 2.5, 2.0, "3. Script Classifier\n(PyTorch CNN)", "5-Class Visual Prior:\nLatin, Arabic, Tamil,\nKannada, Telugu", "#F3E8FF", "#9333EA")

    # Routing Arrows from 3 to Recognizers
    # Branch 1: Latin
    ax.annotate("", xy=(9.0, 6.8), xytext=(8.6, 5.9),
                arrowprops=dict(facecolor="#64748B", edgecolor="#64748B", width=1.5, headwidth=6))
    ax.text(8.7, 6.55, "Latin", fontsize=8.5, fontweight="bold", color="#475569")

    # Branch 2: Indic (Kannada, Tamil, Telugu)
    ax.annotate("", xy=(9.0, 5.4), xytext=(8.6, 5.4),
                arrowprops=dict(facecolor="#64748B", edgecolor="#64748B", width=1.5, headwidth=6))
    ax.text(8.65, 5.15, "Indic Scripts", fontsize=8.5, fontweight="bold", color="#475569")

    # Branch 3: Arabic
    ax.annotate("", xy=(9.0, 4.0), xytext=(8.6, 4.9),
                arrowprops=dict(facecolor="#64748B", edgecolor="#64748B", width=1.5, headwidth=6))
    ax.text(8.7, 4.25, "Arabic", fontsize=8.5, fontweight="bold", color="#475569")

    # Recognizers (Column 4)
    # 4A. Latin + LangID
    draw_box(9.0, 6.2, 2.4, 1.4, "German / Latin OCR", "LangID (German vs Eng)\n+ TrOCR / EasyOCR", "#DCFCE7", "#16A34A")

    # 4B. Shared Indic CRNN
    draw_box(9.0, 4.7, 2.4, 1.4, "Shared Indic CRNN", "7-layer CNN + 2xBiLSTM\n+ Unified 289 Charset\n(Kn, Ta, Te) + CTC Loss", "#FFEDD5", "#EA580C")

    # 4C. Arabic Recognizer
    draw_box(9.0, 3.2, 2.4, 1.4, "Arabic Recognizer", "TrOCR Arabic / EasyOCR\nDirection-aware recognition", "#FCE7F3", "#DB2777")

    # Convergence Arrows to Quality Gate & Translation
    ax.annotate("", xy=(11.8, 5.4), xytext=(11.4, 6.8),
                arrowprops=dict(facecolor="#16A34A", edgecolor="#16A34A", width=1.5, headwidth=6))
    ax.annotate("", xy=(11.8, 5.4), xytext=(11.4, 5.4),
                arrowprops=dict(facecolor="#EA580C", edgecolor="#EA580C", width=1.5, headwidth=6))
    ax.annotate("", xy=(11.8, 5.4), xytext=(11.4, 4.0),
                arrowprops=dict(facecolor="#DB2777", edgecolor="#DB2777", width=1.5, headwidth=6))

    # 5. Quality Gating & Translation
    draw_box(11.8, 4.3, 2.0, 2.2, "5. Translation\n(NLLB-200)", "Quality Gating Check\n↓\nkan/tam/tel/deu/ara\n↓\nEnglish Output", "#E0E7FF", "#4F46E5")

    # Output Box (Bottom)
    draw_box(3.5, 0.8, 7.0, 1.8, "6. Multi-Modal Interactive Demo & Document Inpainting", 
             "• Streamlit UI with Model Selection Toggle (Custom CRNN vs Baseline Standard)\n• Full Preprocessing Ablation Toggles (Sauvola, Deskew, Denoise)\n• Text Inpainting & Typography Overlay on Original Parchment Canvas\n• Live Empirical Benchmarks (CER / WER / Confusion Matrix)", 
             "#F1F5F9", "#0F172A")

    # Arrow from Translation to Demo
    ax.annotate("", xy=(10.5, 2.6), xytext=(12.8, 4.3),
                arrowprops=dict(facecolor="#4F46E5", edgecolor="#4F46E5", width=1.8, headwidth=7))

    plt.tight_layout()
    output_path = "docs/project_architecture_flow.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated {output_path}")

def generate_crnn_diagram():
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)

    ax.text(6, 5.6, "Shared Indic CRNN Recognition Architecture", 
            ha="center", va="center", fontsize=14, fontweight="bold", color="#1E293B")
    ax.text(6, 5.2, "Unified Character Set (Kannada + Tamil + Telugu + Digits + Punctuation + CTC Blank = 289 classes)", 
            ha="center", va="center", fontsize=10, fontstyle="italic", color="#475569")

    layers = [
        ("Input Line/Word Image\n(1 x 32 x W)", "#E0F2FE", "#0284C7"),
        ("CNN Feature Extractor\n7x [Conv2d + BN + ReLU + MaxPool]\nReceptive Field: (512 x 1 x W/4)", "#FEF3C7", "#D97706"),
        ("Bidirectional LSTM\n2-Layer BiLSTM (hidden=256)\nTemporal Sequence Modeling", "#F3E8FF", "#9333EA"),
        ("Linear Projection\nDense(512 -> 289 classes)\n+ Log-Softmax per time frame", "#FFEDD5", "#EA580C"),
        ("CTC Decoding (Greedy)\n1. Frame-wise Argmax\n2. Collapse Consecutive Duplicates\n3. Remove Blank Token (Index 0)", "#DCFCE7", "#16A34A"),
        ("Predicted Text\nKannada / Tamil / Telugu", "#FCE7F3", "#DB2777"),
    ]

    x_start = 0.5
    w = 1.6
    h = 2.8
    y = 1.4

    for i, (text, fcol, ecol) in enumerate(layers):
        x = x_start + i * 1.9
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                                      linewidth=1.6, edgecolor=ecol, facecolor=fcol)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=8.5, fontweight="bold", color="#0F172A")

        if i < len(layers) - 1:
            ax.annotate("", xy=(x + w + 0.3, y + h/2), xytext=(x + w, y + h/2),
                        arrowprops=dict(facecolor="#64748B", edgecolor="#64748B", width=2, headwidth=6))

    plt.tight_layout()
    output_path = "docs/indic_crnn_architecture.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated {output_path}")

if __name__ == "__main__":
    generate_pipeline_diagram()
    generate_crnn_diagram()
