#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python hpi_pdf.py hpi.json \
  --bg assets/hpi_form_page1_trim.png \
  --out out/HPI.pdf \
  --fit shrink

"""

"""
hpi_pdf.py — Render HPI (page 1) PDF from your hpi.json onto a scanned form.

Examples:
  # first pass with guides:
  python hpi_pdf.py hpi.json --bg assets/hpi_form_page1_trim.png --out out/HPI.pdf --show-frames

  # debug a single box (e.g., Allergies):
  python hpi_pdf.py hpi.json --bg assets/hpi_form_page1_trim.png --out out/HPI_only.pdf \
    --show-frames --only "Allergies"

  # once aligned, render final (no guides):
  python hpi_pdf.py hpi.json --bg assets/hpi_form_page1_trim.png --out out/HPI.pdf
"""

import os, json, argparse
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame, Paragraph, KeepInFrame, FrameBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader

PAGE_W, PAGE_H = letter

ORDER = [
    "Chief Complaint / Reason for Consult",
    "Allergies",
    "History of Present Illness",
    "Medications and Dosages",
    "Past Medical/Surgical History",
    "Social History",
    "Family History",
]

def canon(desc: str) -> str:
    d = (desc or "").lower()
    if "chief" in d or "reason for consult" in d: return ORDER[0]
    if "allerg" in d:                                return ORDER[1]
    if "history of present illness" in d:            return ORDER[2]
    if "medication" in d or "dosage" in d:           return ORDER[3]
    if "past medical" in d or "surgical" in d:       return ORDER[4]
    if "social" in d:                                 return ORDER[5]
    if "family" in d:                                 return ORDER[6]
    return desc or "Unlabeled"

# ---- Relative boxes (fractions 0..1 *inside the calibrated content rectangle*) ----
# These are a good starting shape for the image you shared; we’ll nudge if needed.
BOXES_REL = {
    # left, bottom, width, height
    "Chief Complaint / Reason for Consult": (0.048, 0.604, 0.598, 0.176),
    "Allergies":                            (0.655, 0.604, 0.300, 0.176),

    "History of Present Illness":           (0.048, 0.387, 0.598, 0.198),
    "Medications and Dosages":              (0.655, 0.387, 0.300, 0.198),

    "Past Medical/Surgical History":        (0.048, 0.095, 0.598, 0.252),
    "Social History":                       (0.655, 0.200, 0.300, 0.132),
    "Family History":                       (0.655, 0.105, 0.300, 0.090),
}

# ---- Per-box micro nudges (points). Positive dx->right, dy->up; dW/dH resize. ----
NUDGE = {
    # Example tuned guesses for your screenshot; adjust as you test:
    # Right column looked a hair narrow & left → push right & widen slightly
    "Allergies": {"dx": -100, "dW": +120, "dy": +70, "dH": -40},
    "Medications and Dosages": {"dx": -100, "dW": +120, "dy": -20, "dH": +90},
    "Social History": {"dx": -100, "dW": +120},
    "Family History": {"dx": -100, "dW": +120, "dy": -60, "dH": +50},
    "History of Present Illness": {"dx": -30, "dW": -65, "dy": -30, "dH": +100},
    "Chief Complaint / Reason for Consult": {"dx": -30, "dW": -65, "dy": +70, "dH": -40},
    "Past Medical/Surgical History": {"dx": -30, "dW": -65, "dy": -50, "dH": +40},
    "Social History": {"dx": -100, "dW": 120, "dy": +0, "dH": +0},
    "Family History": {"dx": -100, "dW": 120, "dy": -65, "dH": +50},
}

# ---- Styles (content only; labels live on the background form) ----
_styles = getSampleStyleSheet()
BODY = ParagraphStyle(
    "BoxBody",
    parent=_styles["Normal"],
    fontName="Helvetica",
    fontSize=10,
    leading=12,
    textColor=colors.black,
)
NOTES = ParagraphStyle(
    "Notes",
    parent=_styles["Italic"],
    fontSize=9,
    leading=11,
    textColor=colors.black,
)

def load_hpi(json_path: str):
    data = json.load(open(json_path, "r", encoding="utf-8"))
    sections = {k: {"body": "", "notes": ""} for k in ORDER}
    for _, item in sorted(data.items()):
        sec  = canon(item.get("Description", ""))
        body = (item.get("Response") or "").strip()
        note = (item.get("Notes") or "").strip()
        if sec not in sections:
            sections[sec] = {"body": "", "notes": ""}
        if body:
            sections[sec]["body"] += (("\n\n" if sections[sec]["body"] else "") + body)
        if note:
            sections[sec]["notes"] += (("\n" if sections[sec]["notes"] else "") + note)
    return sections

def to_abs_rel(l, b, w, h, cal_left, cal_right, cal_top, cal_bottom):
    """Map relative box into calibrated inner rectangle."""
    x0 = cal_left   * PAGE_W
    x1 = (1.0 - cal_right)  * PAGE_W
    y0 = cal_bottom * PAGE_H
    y1 = (1.0 - cal_top)    * PAGE_H
    CW, CH = (x1 - x0), (y1 - y0)
    return (x0 + l * CW, y0 + b * CH, w * CW, h * CH)

def on_page_draw(bg_path: str, show_frames: bool, cal, only=None):
    cal_left, cal_right, cal_top, cal_bottom = cal
    def _draw(canvas: Canvas, doc):
        if bg_path and os.path.exists(bg_path):
            canvas.drawImage(ImageReader(bg_path), 0, 0, width=PAGE_W, height=PAGE_H,
                             preserveAspectRatio=True, mask='auto')
        if show_frames:
            sections = [only] if only else ORDER
            canvas.saveState()
            canvas.setStrokeColor(colors.red)
            for sec in sections:
                l,b,w,h = BOXES_REL[sec]
                x,y,W,H = to_abs_rel(l,b,w,h, cal_left, cal_right, cal_top, cal_bottom)
                adj = NUDGE.get(sec, {})
                x += adj.get("dx", 0); y += adj.get("dy", 0)
                W += adj.get("dW", 0); H += adj.get("dH", 0)
                canvas.rect(x, y, W, H, stroke=1, fill=0)
            canvas.restoreState()
    return _draw

def render_pdf(hpi_json: str, out_pdf: str, bg_image: str,
               fit_mode: str, show_frames: bool,
               cal_left: float, cal_right: float, cal_top: float, cal_bottom: float,
               only: str | None):
    os.makedirs(os.path.dirname(out_pdf) or ".", exist_ok=True)
    sections = load_hpi(hpi_json)

    section_list = [only] if only else ORDER

    # frames
    frames = []
    for sec in section_list:
        l,b,w,h = BOXES_REL[sec]
        x,y,W,H = to_abs_rel(l,b,w,h, cal_left, cal_right, cal_top, cal_bottom)
        adj = NUDGE.get(sec, {})
        x += adj.get("dx", 0); y += adj.get("dy", 0)
        W += adj.get("dW", 0); H += adj.get("dH", 0)

        frames.append(Frame(
            x, y, W, H,
            leftPadding=6, rightPadding=6, topPadding=4, bottomPadding=4,
            showBoundary=0
        ))

    doc = BaseDocTemplate(
        out_pdf, pagesize=letter,
        leftMargin=0, rightMargin=0, topMargin=0, bottomMargin=0,
        pageTemplates=[PageTemplate(
            id="form",
            frames=frames,
            onPage=on_page_draw(bg_image, show_frames, (cal_left, cal_right, cal_top, cal_bottom), only=only)
        )]
    )

    story = []
    for i, sec in enumerate(section_list):
        body  = sections.get(sec, {}).get("body") or "—"
        notes = sections.get(sec, {}).get("notes") or ""

        flow = [Paragraph(body.replace("\n", "<br/>"), BODY)]
        if notes:
            safe = notes.replace("<","&lt;").replace(">","&gt;").replace("\n","<br/>")
            flow.append(Paragraph(f"<i>Notes:</i> {safe}", NOTES))

        if fit_mode == "shrink":
            story.append(KeepInFrame(0, 0, content=flow, mode="shrink"))
        elif fit_mode == "truncate":
            story.append(KeepInFrame(0, 0, content=flow, mode="truncate"))
        else:
            story.extend(flow)

        if i < len(section_list) - 1:
            story.append(FrameBreak())

    doc.build(story)
    print(f"[hpi_pdf] wrote: {out_pdf}")

def main():
    ap = argparse.ArgumentParser(description="Render HPI page PDF from hpi.json")
    ap.add_argument("json", help="Path to hpi.json")
    ap.add_argument("--bg", required=True, help="Background image (trimmed to the outer frame is best)")
    ap.add_argument("--out", default="out/HPI.pdf", help="Output PDF path")
    ap.add_argument("--fit", choices=["shrink","truncate","overflow"], default="shrink",
                    help="Fit content inside each box (default: shrink)")
    ap.add_argument("--show-frames", action="store_true", help="Draw red alignment rectangles")

    # Small, symmetric defaults (assumes you trimmed the PNG)
    ap.add_argument("--cal-left", type=float, default=0.010, help="Fraction of page width to trim from LEFT")
    ap.add_argument("--cal-right", type=float, default=0.010, help="Fraction of page width to trim from RIGHT")
    ap.add_argument("--cal-top", type=float, default=0.008, help="Fraction of page height to trim from TOP")
    ap.add_argument("--cal-bottom", type=float, default=0.008, help="Fraction of page height to trim from BOTTOM")

    ap.add_argument("--only",
        choices=ORDER,
        help="Render/debug only this box (draws just one frame and flows only its content)")

    args = ap.parse_args()

    render_pdf(
        args.json, args.out, args.bg, args.fit, args.show_frames,
        args.cal_left, args.cal_right, args.cal_top, args.cal_bottom,
        args.only
    )

if __name__ == "__main__":
    main()
