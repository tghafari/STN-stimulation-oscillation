"""Shared participant-PDF helpers for the all-channel EEG pipeline."""
from __future__ import annotations
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ANALYSIS_DIR = HERE.parents[1]
UTILS_DIR = ANALYSIS_DIR / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))
from pdf_report import ParticipantPDF  # noqa: E402


def participant_report(project_root: Path, subject: str) -> ParticipantPDF:
    folder = project_root / "derivatives" / "reports" / f"sub-{subject}"
    return ParticipantPDF(str(folder), subject)


def figure_dir(project_root: Path, subject: str) -> Path:
    out = project_root / "derivatives" / "figures" / f"sub-{subject}" / "EEG_all_channels"
    out.mkdir(parents=True, exist_ok=True)
    return out


def fmt_channels(channels) -> str:
    values = [str(ch) for ch in channels]
    return ", ".join(values) if values else "None"


def remove_old_posterior_channel_quality_section(report: ParticipantPDF) -> None:
    """Remove only the old posterior 'Epoching and channel quality' section."""
    kept = [item for item in report.items if item.get("section") != "Epoching and channel quality"]
    if len(kept) == len(report.items):
        return
    report.items = kept
    report.manifest_fname.write_text(
        json.dumps(kept, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report._build_pdf()
