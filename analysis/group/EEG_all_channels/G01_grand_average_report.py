#!/usr/bin/env python
"""Deprecated legacy all-channel group report.

Use run_grand_average.py (which runs G04_complete_grand_average_report.py).
This file intentionally does not create a second PDF under
EEG_all_channels_grand_average. It remains only to give a clear migration message
for old commands/scripts.
"""
raise SystemExit(
    "G01_grand_average_report.py is deprecated and no longer writes a separate report. "
    "Run: python run_grand_average.py --subjects <subject IDs>"
)
