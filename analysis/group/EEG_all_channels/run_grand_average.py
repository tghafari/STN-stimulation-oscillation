#!/usr/bin/env python
"""Run the complete all-channel group grand-average report.

This entry point intentionally contains NO monkey-patching or proxying of MNE.
All plotting behavior now lives in G04_complete_grand_average_report.py itself.
Keeping the runner this small makes `%run run_grand_average.py ...` safe to execute
repeatedly in IPython/Jupyter without leaving modified MNE functions in memory.
"""
from G04_complete_grand_average_report import main

if __name__ == "__main__":
    main()
