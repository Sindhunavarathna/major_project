"""Generate publication-grade System Architecture Diagram for Major Project."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def create_architecture_diagram(output_path: str = "docs/architecture_diagram.png"):
    # High resolution 16:9 canvas
    fig, ax = plt.subplots(figsize=(20, 11), dpi=300)
    fig.patch.set_facecolor('#0B0F19')
    ax.set_facecolor('#0B0F19')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis('off')

    # Color Palette (Dark Theme / Engineering Blueprint)
    C_HEADER = '#38BDF8'
    C_BORDER = '#1E293B'
    C_CARD_BG = '#111827'
    C_STAGE_BG = '#1E293B'
    C_TEXT_LIGHT = '#F8FAFC'
    C_TEXT_MUTED = '#94A3B8'
    C_ACCENT_BLUE = '#0284C7'
    C_ACCENT_PURPLE = '#8B5CF6'
    C_ACCENT_EMERALD = '#10B981'
    C_ACCENT_AMBER = '#F59E0B'
    C_ACCENT_ROSE = '#F43F5E'

    # Title Banner
    ax.text(50, 56.5, "AI-Powered Multilingual Manuscript OCR & Translation System",
            ha='center', va='center', fontsize=21, fontweight='bold', color=C_TEXT_LIGHT, fontfamily='sans-serif')
    ax.text(50, 54.2, "End-to-End Paleographic Preprocessing, Multi-Script Vision Routing, LID & 4-Tier Hybrid Translation Matrix",
            ha='center', va='center', fontsize=12, color=C_HEADER, fontfamily='sans-serif')

    # Helper function for drawing rounded styled cards
    def draw_card(x, y, w, h, title, subtitle="", badge="", badge_color=C_ACCENT_BLUE, border_color=C_BORDER, fill_color=C_CARD_BG):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.6,rounding_size=1.2",
            facecolor=fill_color,
            edgecolor=border_color,
            linewidth=1.8,
            zorder=2
        )
        ax.add_patch(rect)
        if badge:
            badge_box = patches.FancyBboxPatch(
                (x + 1.2, y + h - 2.2), len(badge)*0.75 + 1.2, 1.6,
                boxstyle="round,pad=0.2,rounding_size=0.6",
                facecolor=badge_color,
                edgecolor='none',
                zorder=3
            )
            ax.add_patch(badge_box)
            ax.text(x + 1.8, y + h - 1.4, badge, fontsize=8, fontweight='bold', color='#FFFFFF', va='center', zorder=4)

        title_y = y + h - 1.4 if not badge else y + h - 1.4
        title_x = x + 1.5 if not badge else x + len(badge)*0.75 + 3.2
        ax.text(title_x, title_y, title, fontsize=12, fontweight='bold', color=C_TEXT_LIGHT, va='center', zorder=4)
        if subtitle:
            ax.text(x + 1.5, y + h - 3.2, subtitle, fontsize=9, color=C_TEXT_MUTED, va='center', zorder=4)

    def draw_bullet(x, y, label, desc="", color=C_TEXT_LIGHT):
        ax.plot(x, y, marker='o', markersize=4, color=C_ACCENT_BLUE, zorder=5)
        ax.text(x + 1.0, y, label, fontsize=9.5, fontweight='bold', color=color, va='center', zorder=5)
        if desc:
            ax.text(x + 1.0, y - 1.3, desc, fontsize=8.2, color=C_TEXT_MUTED, va='center', zorder=5)

    def draw_arrow(x1, y1, x2, y2, label="", color='#475569', style='->', lw=2.0, curved=0.0):
        connectionstyle = f"arc3,rad={curved}" if curved != 0.0 else "arc3,rad=0"
        arrow = patches.FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle=style,
            color=color,
            linewidth=lw,
            connectionstyle=connectionstyle,
            mutation_scale=15,
            zorder=3
        )
        ax.add_patch(arrow)
        if label:
            mx, my = (x1 + x2)/2, (y1 + y2)/2 + 1.0
            ax.text(mx, my, label, fontsize=8.5, fontweight='bold', color=C_HEADER,
                    ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='#0B0F19', edgecolor=color, linewidth=1),
                    zorder=6)

    # -------------------------------------------------------------
    # STAGE 1: INGESTION & PREPROCESSING
    # -------------------------------------------------------------
    draw_card(2, 6, 21, 44, "STAGE 1: DOCUMENT INGESTION", "Paleographic Preprocessing Pipeline", "INPUT", C_ACCENT_PURPLE)
    
    # Sub-box 1.1: Input Manuscripts
    draw_card(3.5, 36, 18, 9.5, "Source Manuscripts", "Historical Substrates", "", fill_color='#1E293B')
    draw_bullet(4.5, 41.5, "Script Coverage", "Arabic, Sanskrit, Kannada, Hindi")
    draw_bullet(4.5, 38.5, "Substrates", "Palm-leaf, Parchment, Epigraphs")

    # Sub-box 1.2: Image Preprocessing
    draw_card(3.5, 7.5, 18, 27, "Image Enhancement", "OpenCV & NumPy Algorithms", "", fill_color='#1E293B')
    draw_bullet(4.5, 30.5, "Adaptive Contrast", "Mild CLAHE & Illumination Normalize")
    draw_bullet(4.5, 26.5, "Denoising", "Bilateral filter preserving ink strokes")
    draw_bullet(4.5, 22.5, "Deskew Correction", "Hough Transform & MinAreaRect (±45°)")
    draw_bullet(4.5, 18.5, "Segmentation", "CRAFT line & token box grouping")
    draw_bullet(4.5, 14.5, "Substrate Calibration", "Adaptive thresholding without bleed")
    draw_bullet(4.5, 10.5, "Dual Outputs", "Enhanced Grayscale + Clean Binarized")

    # Arrow Stage 1 -> Stage 2
    draw_arrow(23.5, 28, 26.5, 28, "Processed Lines", C_ACCENT_PURPLE, lw=2.5)

    # -------------------------------------------------------------
    # STAGE 2: VISION OCR & QUALITY GATING
    # -------------------------------------------------------------
    draw_card(27, 6, 22, 44, "STAGE 2: SCRIPT OCR & QUALITY", "Dynamic Model Router & Calibration", "VISION", C_ACCENT_BLUE)

    # Sub-box 2.1: Dynamic Router
    draw_card(28.5, 25, 19, 20.5, "Vision OCR Router", "Script-Specific Pure Ensembles", "", fill_color='#1E293B')
    draw_bullet(29.5, 41.0, "Arabic Pure Engine", "EasyOCR ['ar'] (Naskh/Thuluth font calibration)")
    draw_bullet(29.5, 36.8, "Kannada Epigraph Engine", "EasyOCR ['kn'] (Halegannada glyph coverage)")
    draw_bullet(29.5, 32.6, "Devanagari Engine", "EasyOCR ['hi'] (Sanskrit/Hindi ligatures)")
    draw_bullet(29.5, 28.4, "Zero Latin Pollution", "Single-script models prevent English hallucinations")

    # Sub-box 2.2: Quality Gating
    draw_card(28.5, 7.5, 19, 16, "Quality Gating & Scoring", "Confidence Calibration & Verification", "", fill_color='#1E293B')
    draw_bullet(29.5, 19.5, "Weighted Scoring", "Length-weighted paleographic confidence")
    draw_bullet(29.5, 15.5, "Threshold Gate", "Q_thresh >= 25% (Warning on corrupt noise)")
    draw_bullet(29.5, 11.5, "Paleographic Postproc", "Halant, Nukta & Diacritic normalization")

    # Arrow Stage 2 -> Stage 3
    draw_arrow(49.5, 28, 52.5, 28, "Clean Tokens", C_ACCENT_BLUE, lw=2.5)

    # -------------------------------------------------------------
    # STAGE 3: LANGUAGE IDENTIFICATION (LID)
    # -------------------------------------------------------------
    draw_card(53, 31, 20, 19, "STAGE 3: LANGUAGE ID", "Hierarchical Script & Language Classifier", "LID", C_ACCENT_AMBER)
    draw_bullet(54.5, 45.0, "Fast Unicode Block Check", "Zero-latency script isolation (Dev/Kan/Arab)")
    draw_bullet(54.5, 40.5, "Character N-Gram NB", "Multinomial Naive Bayes (99.98% Accuracy)")
    draw_bullet(54.5, 36.0, "Confidence Calibration", "Softmax posterior with entropy verification")

    # Arrow Stage 3 -> Stage 4
    draw_arrow(63, 31, 63, 26, "Verified Lang Code", C_ACCENT_AMBER, lw=2.5)

    # -------------------------------------------------------------
    # STAGE 4: HYBRID TRANSLATION MATRIX
    # -------------------------------------------------------------
    draw_card(53, 6, 20, 19, "STAGE 4: TRANSLATION MATRIX", "4-Tier Classical & Neural Bridge", "HYBRID", C_ACCENT_EMERALD)
    draw_bullet(54.5, 20.5, "Tier 1: Canonical Resolver", "Met 21.148.6, Nicene Creed, Quran 2:148, Shlokas")
    draw_bullet(54.5, 16.5, "Tier 2: Neural Cloud Bridge", "Google Deep-Translator + Hallucination Guard")
    draw_bullet(54.5, 12.5, "Tier 3: Classical Lexicons", "Halegannada (Old Kannada) & Sanskrit Morph engine")
    draw_bullet(54.5, 8.5, "Tier 4: Local Neural Model", "mBART-50 Seq2Seq (Offline Transformer Fallback)")

    # Arrow Stage 4 -> Stage 5
    draw_arrow(73.5, 15.5, 76.5, 15.5, "Validated Translation", C_ACCENT_EMERALD, lw=2.5)
    draw_arrow(40, 7.5, 86, 7.5, "Bounding Boxes (Line Coordinates)", '#475569', lw=1.5, curved=-0.2)

    # -------------------------------------------------------------
    # STAGE 5: POSTPROCESSING & OUTPUT STUDIO
    # -------------------------------------------------------------
    draw_card(77, 6, 21, 44, "STAGE 5: USER STUDIO & OUTPUT", "Inpainting, UI & Document Generation", "OUTPUT", C_ACCENT_ROSE)

    # Sub-box 5.1: Inpainting Canvas
    draw_card(78.5, 27.5, 18, 18, "Inpainting & Layout Engine", "Seamless Document Reconstruction", "", fill_color='#1E293B')
    draw_bullet(79.5, 41.5, "Background Sampling", "Substrate border pixel interpolation")
    draw_bullet(79.5, 37.0, "Text Region Inpainting", "Telea / Navier-Stokes mask reconstruction")
    draw_bullet(79.5, 32.5, "Dynamic Typography", "Auto-scaling font bounding box fit")

    # Sub-box 5.2: Streamlit Studio & Export
    draw_card(78.5, 7.5, 18, 18, "Dual-Panel Studio UI", "Executive Minimalist Dashboard", "", fill_color='#1E293B')
    draw_bullet(79.5, 21.5, "Dual Studio Panels", "Source Document vs Translation Output")
    draw_bullet(79.5, 17.0, "Editable Transcription", "Human-in-the-loop expert review")
    draw_bullet(79.5, 12.5, "One-Click Export", "High-res Inpainted PNG & Text (.TXT)")

    # Footer Metadata
    ax.text(50, 2.0, "Engineering Major Project Architecture • Antigravity AI • Python 3.12 • PyTorch • EasyOCR • mBART-50 • OpenCV • Streamlit",
            ha='center', va='center', fontsize=9.5, color='#64748B', fontfamily='sans-serif')

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Diagram successfully generated at: {output_path}")

if __name__ == "__main__":
    create_architecture_diagram("docs/system_architecture.png")
    create_architecture_diagram("docs/architecture_diagram.png")
