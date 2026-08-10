# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
===============================================
04_segmenting_stim
    1. this code reads the BIDS raw file
    2. crops the continuous recording into stim on
    and stim off sections
    3. removes the part in between the kept segments
    4. plots the PSD for each part separately to
    double check if the 130Hz peak exists or not
    5. saves the cropped stim and no-stim files
    6. adds the segmentation details and PSD figures
    to the participant PDF report

    note that the crop times are kept in the code
    for each participant, because these values are
    checked manually from the raw data.

written by Tara Ghafari
tara.ghafari@gmail.com
adapted from flux pipeline
==============================================
"""

import os
import os.path as op
import sys
import mne
from mne_bids import BIDSPath, read_raw_bids

GITHUB_ROOT = r'/Users/taraghafari/Desktop/Desktop - Tara’s MacBook Pro/BEAR_outage/GitHub/STN-stimulation-oscillation'
UTILS_DIR = os.path.join(GITHUB_ROOT, 'analysis', 'utils')

if GITHUB_ROOT not in sys.path:
    sys.path.insert(0, GITHUB_ROOT)
if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)

from pdf_report import ParticipantPDF

subject = '117'
session = '01'
task = 'SpAtt'
run = '01'
eeg_suffix = 'eeg'
extension = '.fif'
project_root = '/Users/taraghafari/Desktop/Desktop - Tara’s MacBook Pro/BEAR_outage/STN-in-PD'  # local folder
bids_root = op.join(project_root, 'data', 'BIDS')
bids_path = BIDSPath(subject=subject, session=session, task=task, run=run,
                     root=bids_root, datatype='eeg', suffix=eeg_suffix)
deriv_folder = op.join(bids_root, 'derivatives', 'sub-' + subject)
fig_folder = op.join(project_root, 'derivatives', 'figures', f'sub-{subject}')
report_folder = op.join(project_root, 'derivatives', 'reports', f'sub-{subject}')
os.makedirs(deriv_folder, exist_ok=True)
os.makedirs(fig_folder, exist_ok=True)
report = ParticipantPDF(report_folder, subject)

raw = read_raw_bids(bids_path=bids_path, verbose=True, extra_params={'preload': True})
raw.plot()  # visual confirmation of your saved crop times

# Keep and extend your existing crop dictionary. Four numbers mean two kept pieces;
# the interval between the pieces is not included in the concatenated output.
stimulation_cropped_time = {
    'sub-107_no-stim': [15, 974], 'sub-107_stim': [1000, 1845],
    'sub-108_no-stim': [8, 890], 'sub-108_stim': [930, 1882],
    'sub-110_no-stim': [905, 1711], 'sub-110_stim': [0, 840],
    'sub-102_no-stim': [0, 965], 'sub-102_stim': [1490, 2230],
    'sub-101_no-stim': [0, 360, 515, 865], 'sub-101_stim': [1144, 1900],
    'sub-112_no-stim': [878, 1289, 1870, 2280], 'sub-112_stim': [244, 650, 2708, 3117],
    'sub-103_no-stim': [785, 1113, 1380, 1715], 'sub-103_stim': [72, 476, 1862, 2182],
    'sub-104_no-stim': [1100, 1412, 1946, 2269], 'sub-104_stim': [9, 772],
    'sub-105_no-stim': [4326, 5103], 'sub-105_stim': [112, 597, 850, 1327],
    'sub-113_no-stim': [144, 507, 883, 1233], 'sub-113_stim': [1895, 2255, 2306, 2668],
    'sub-115_no-stim': [1523, 1960, 2353, 2750], 'sub-115_stim': [30, 620, 768, 1235],
}
# sub 115 stim sequence does not follow the stim sequence in the table on github. so I 
# I can't know which no stim segment has right lfp rec and which has left lfp rec. but i can tell 
# from the time series that first half is stim on second half is stim off.

def make_segment(label):
    times = stimulation_cropped_time[f'sub-{subject}_{label}']
    if len(times) not in (2, 4):
        raise ValueError(f'{label} must contain 2 or 4 crop times, got {times}')
    pieces = [raw.copy().crop(tmin=times[0], tmax=times[1])]
    if len(times) == 4:
        pieces.append(raw.copy().crop(tmin=times[2], tmax=times[3]))
    # Concatenation excludes the section between the retained pieces.
    segment = pieces[0] if len(pieces) == 1 else mne.concatenate_raws(pieces)
    return segment, times

for label in ['no-stim', 'stim']:
    suffix = label
    segment, times = make_segment(label)
    # PSD before the 100-Hz low-pass, so the 130-Hz stimulation peak remains visible.
    fmax = min(200, segment.info['sfreq'] / 2 - 0.1)
    fig_psd = segment.compute_psd(fmin=0.1, fmax=fmax).plot(show=False)
    report.add_figure(fig_psd, op.join(fig_folder, f'P04_{label}_PSD_before_filter.png'),
                      f'{label} PSD before filtering or any other processing.',
                      f'Used to check whether a peak near 130 Hz is present. Kept ranges: {times}',
                      'Stimulation segmentation')

    # Apply the requested analysis filter only after the 130-Hz QC PSD.
    segment.filter(l_freq=0.1, h_freq=100.0)
    output = op.join(deriv_folder, bids_path.basename + f'_{suffix}_raw.fif')
    segment.save(output, overwrite=True)
    report.add_text(f'{label} segment saved',
                    f'Kept ranges: {times}\nFiltered 0.1-100 Hz\nOutput: {output}',
                    'Stimulation segmentation')

print(f'Updated PDF: {report.pdf_fname}')
