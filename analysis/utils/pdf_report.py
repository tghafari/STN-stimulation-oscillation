"""Persistent ReportLab report helper for the STN EEG scripts.

Each analysis script appends short notes and saved figures to a JSON
manifest. The PDF is rebuilt from that manifest, so later steps can keep
adding content to one participant-level report instead of replacing earlier
pages.

The helper can read channel impedance values directly from BrainVision .vhdr
header files, which is useful when the information is stored in the recording
header rather than exposed through an already-loaded MNE object.

Requires: pip install reportlab pillow
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer
from reportlab.platypus.flowables import HRFlowable


_IMPEDANCE_START_RE = re.compile(r"^Impedance\s+\[[^\]]+\]\s+at\s+\d\d:\d\d:\d\d\s*:\s*$")
_IMPEDANCE_LINE_RE = re.compile(r"^[ A-Za-z0-9_+\-]+:\s*.*$")
_NUMBER_RE = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")


class ParticipantPDF:
    def __init__(self, report_folder: str, subject: str):
        self.subject = str(subject)
        self.folder = Path(report_folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.manifest_fname = self.folder / f"sub-{self.subject}_report_manifest.json"
        self.pdf_fname = self.folder / f"sub-{self.subject}_analysis_report.pdf"
        if self.manifest_fname.exists():
            self.items = json.loads(self.manifest_fname.read_text(encoding="utf-8"))
        else:
            self.items = []

    def add_text(self, title: str, text: str, section: str = "General") -> None:
        self.items.append({
            "kind": "text", "section": section, "title": title,
            "text": str(text), "created": datetime.now().isoformat(timespec="seconds")
        })
        self._save_and_build()

    def add_figure(self, fig, image_fname: str, title: str,
                   caption: str = "", section: str = "General",
                   dpi: int = 180) -> None:
        image_fname = str(image_fname)
        Path(image_fname).parent.mkdir(parents=True, exist_ok=True)
        if hasattr(fig, "savefig"):
            fig.savefig(image_fname, dpi=dpi, bbox_inches="tight")
        elif isinstance(fig, (list, tuple)) and fig and hasattr(fig[0], "savefig"):
            fig[0].savefig(image_fname, dpi=dpi, bbox_inches="tight")
        else:
            raise TypeError("fig must be a Matplotlib/MNE figure with savefig().")
        self.add_image(image_fname, title, caption, section)

    def add_image(self, image_fname: str, title: str,
                  caption: str = "", section: str = "General") -> None:
        image_fname = os.path.abspath(image_fname)
        if not os.path.exists(image_fname):
            raise FileNotFoundError(image_fname)
        self.items.append({
            "kind": "image", "section": section, "title": title,
            "path": image_fname, "caption": caption,
            "created": datetime.now().isoformat(timespec="seconds")
        })
        self._save_and_build()

    def add_key_values(self, title: str, values: dict,
                       section: str = "General") -> None:
        text = "\n".join(f"{key}: {value}" for key, value in values.items())
        self.add_text(title, text, section)

    def _save_and_build(self) -> None:
        self.manifest_fname.write_text(
            json.dumps(self.items, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self._build_pdf()

    def _build_pdf(self) -> None:
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "TitleStyle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            spaceAfter=8,
        )
        section_style = ParagraphStyle(
            "SectionStyle",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=17,
            textColor=colors.HexColor("#1f3c88"),
            spaceBefore=6,
            spaceAfter=6,
        )
        item_title_style = ParagraphStyle(
            "ItemTitleStyle",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=13,
            spaceBefore=4,
            spaceAfter=3,
        )
        body_style = ParagraphStyle(
            "BodyStyle",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=11,
            spaceAfter=3,
            alignment=TA_LEFT,
        )
        caption_style = ParagraphStyle(
            "CaptionStyle",
            parent=styles["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#444444"),
            spaceAfter=6,
        )
        meta_style = ParagraphStyle(
            "MetaStyle",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#666666"),
            spaceAfter=6,
        )

        def _para(text, style):
            return Paragraph(escape(str(text)).replace("\n", "<br/>") , style)

        def _image_flowable(path):
            try:
                pil_img = PILImage.open(path)
                width_px, height_px = pil_img.size
                pil_img.close()
            except Exception:
                width_px, height_px = 1000, 700

            max_w_pt = 175 * mm
            max_h_pt = 170 * mm
            dpi_assumed = 180.0
            w_pt = width_px / dpi_assumed * 72.0
            h_pt = height_px / dpi_assumed * 72.0
            scale = min(max_w_pt / w_pt, max_h_pt / h_pt, 1.0)
            return Image(path, width=w_pt * scale, height=h_pt * scale)

        story = []
        story.append(Paragraph(f"EEG analysis report: subject {self.subject}", title_style))
        story.append(Paragraph(f"Updated {datetime.now().strftime('%Y-%m-%d %H:%M')}", meta_style))
        story.append(HRFlowable(width="100%", thickness=0.8, color=colors.grey))
        story.append(Spacer(1, 4 * mm))

        previous_section = None
        for item in self.items:
            section = item.get("section", "General")
            if section != previous_section:
                if previous_section is not None:
                    story.append(PageBreak())
                story.append(Paragraph(escape(str(section)), section_style))
                story.append(Spacer(1, 2 * mm))
                previous_section = section

            story.append(Paragraph(escape(str(item.get("title", ""))), item_title_style))
            if item["kind"] == "text":
                story.append(_para(item.get("text", ""), body_style))
            else:
                path = item["path"]
                if os.path.exists(path):
                    story.append(_image_flowable(path))
                caption = item.get("caption", "")
                if caption:
                    story.append(_para(caption, caption_style))
            story.append(Spacer(1, 2 * mm))

        doc = SimpleDocTemplate(
            str(self.pdf_fname),
            pagesize=A4,
            leftMargin=15 * mm,
            rightMargin=15 * mm,
            topMargin=15 * mm,
            bottomMargin=15 * mm,
            title=f"EEG analysis report: subject {self.subject}",
        )
        doc.build(story)


def _extract_impedance_dict_from_vhdr(vhdr_path: str) -> dict:
    """Parse BrainVision impedance values directly from a .vhdr header file."""
    path = Path(vhdr_path)
    if not path.exists():
        raise FileNotFoundError(path)

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    impedances = {}
    for idx, line in enumerate(lines):
        if not _IMPEDANCE_START_RE.match(line.strip()):
            continue
        for setting in lines[idx + 1:]:
            stripped = setting.strip()
            if not stripped or not _IMPEDANCE_LINE_RE.match(setting):
                break
            channel, _, rest = setting.partition(":")
            channel = channel.strip()
            nums = _NUMBER_RE.findall(rest)
            imp_value = float(nums[0]) if nums else None
            impedances[channel] = {"imp": imp_value, "imp_unit": "kOhm"}
        break
    return impedances


def _format_impedance_entries(impedances: Mapping[str, object]) -> str:
    if not impedances:
        return "No impedance values were found in the BrainVision header."

    lines = []
    for channel, value in impedances.items():
        if isinstance(value, Mapping):
            imp = value.get("imp", value.get("impedance", value))
            unit = value.get("imp_unit", value.get("unit", ""))
            unit_txt = f" {unit}" if unit else ""
            lines.append(f"{channel}: {imp}{unit_txt}")
        else:
            lines.append(f"{channel}: {value}")
    return "\n".join(lines)


def impedance_text(raw=None, vhdr_path: str | Sequence[str] | None = None) -> str:
    """Return readable BrainVision impedance information.

    The function first tries the provided .vhdr path(s). If nothing is found,
    it falls back to raw.impedances when available.
    """
    paths = []
    if vhdr_path is not None:
        if isinstance(vhdr_path, (list, tuple, set)):
            paths.extend([str(p) for p in vhdr_path])
        else:
            paths.append(str(vhdr_path))

    combined = {}
    for path in paths:
        try:
            combined.update(_extract_impedance_dict_from_vhdr(path))
        except Exception:
            continue

    if combined:
        return _format_impedance_entries(combined)

    impedances = getattr(raw, "impedances", None)
    if impedances:
        return _format_impedance_entries(impedances)

    return "No impedance values were available in the BrainVision header."
