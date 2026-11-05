"""Helpers for participant-level PDF reports in the all-channel pipeline.

The participant report is shared with posterior-channel preprocessing.
Figures are written to a subject-specific derivative directory, while
the report manifest tracks sections and their associated figures.
"""

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
    """Return the persistent PDF report for one participant."""
    folder = project_root / "derivatives" / "reports" / f"sub-{subject}"
    return ParticipantPDF(str(folder), subject)


def figure_dir(project_root: Path, subject: str) -> Path:
    """Create the participant's all-channel figure output directory."""
    output = (
        project_root / "derivatives" / "figures"
        / f"sub-{subject}" / "EEG_all_channels"
    )
    output.mkdir(parents=True, exist_ok=True)
    return output


def fmt_channels(channels) -> str:
    """Format channel names for human-readable QC reports."""
    values = [str(channel) for channel in channels]
    return ", ".join(values) if values else "None"


def remove_old_posterior_channel_quality_section(
    report: ParticipantPDF,
) -> None:
    """Remove only the obsolete posterior channel-quality report section.

    This preserves every other report section and regenerates the PDF
    only if the old section was actually present in the manifest.
    """
    kept = [
        item for item in report.items
        if item.get("section") != "Epoching and channel quality"
    ]
    if len(kept) == len(report.items):
        return

    report.items = kept
    report.manifest_fname.write_text(
        json.dumps(kept, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report._build_pdf()
