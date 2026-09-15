#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Single all-subject posterior TFR QC report.

ONE PDF contains ALL requested subjects. The report is organised by result. For each
result, every subject is shown sequentially with exactly two within-subject plots:
mean ROI8 and mean ROI3. There is no across-subject averaging.
"""
from G03_participant_eight_posterior_TFR_QC import main

if __name__ == '__main__':
    main()
