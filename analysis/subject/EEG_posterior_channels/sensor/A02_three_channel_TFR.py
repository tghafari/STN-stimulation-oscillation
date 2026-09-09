# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
===============================================
A02_three_channel_TFR
    1. this code reads the cleaned cue epochs
    2. combines attention right and attention left
    epochs together for each stimulation condition
    3. calculates the TFR separately for stim on and
    stim off
    4. compares stim on minus stim off for the three
    posterior channels
    5. adds the TFR figures and saved outputs to the
    participant PDF report

    note that the analysis is restricted to the same
    three posterior channels used for manual epoch
    rejection, so the comparison stays consistent
    across the whole pipeline.

written by Tara Ghafari
tara.ghafari@gmail.com
==============================================

"""

import json
import os
import os.path as op
import sys

import numpy as np
import matplotlib.pyplot as plt
import mne
from mne_bids import BIDSPath

GITHUB_ROOT = r'/Users/taraghafari/Desktop/Desktop - Tara’s MacBook Pro/BEAR_outage/GitHub/STN-stimulation-oscillation'
UTILS_DIR = os.path.join(GITHUB_ROOT, 'analysis', 'utils')

if GITHUB_ROOT not in sys.path:
    sys.path.insert(0, GITHUB_ROOT)
if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)

from pdf_report import ParticipantPDF

subject = '115'
session = '01'
task = 'SpAtt'
run = '01'
eeg_suffix = 'eeg'
project_root = '/Users/taraghafari/Desktop/Desktop - Tara’s MacBook Pro/BEAR_outage/STN-in-PD'  # local folder
bids_root = op.join(project_root, 'data', 'BIDS')
posterior_file = op.join(
    bids_root,
    "derivatives",
    f"sub-{subject}",
    "qc",
    f"sub-{subject}_posterior_channels.json",
)

if op.exists(posterior_file):
    with open(posterior_file, "r", encoding="utf-8") as f:
        posterior_channels = json.load(f)
else:
    posterior_channels = ['PO3', 'PO4', 'POz']


baseline = (-0.3, -0.1)

bids_path = BIDSPath(subject=subject, session=session, task=task, run=run,
                     root=bids_root, datatype='eeg', suffix=eeg_suffix)
deriv_folder = op.join(bids_root, 'derivatives', 'sub-' + subject)
fig_folder = op.join(project_root, 'derivatives', 'figures', f'sub-{subject}')
report_folder = op.join(project_root, 'derivatives', 'reports', f'sub-{subject}')
os.makedirs(fig_folder, exist_ok=True)
report = ParticipantPDF(report_folder, subject)

freqs = np.arange(2, 31, 1)
n_cycles = freqs / 2

tfrs_raw = {}
tfrs_plot = {}

for label in ['no-stim', 'stim']:
    input_fname = op.join(deriv_folder, bids_path.basename + f'_{label}_epo-cue.fif')
    epochs = mne.read_epochs(input_fname, preload=True)

    missing = [ch for ch in posterior_channels if ch not in epochs.ch_names]
    if missing:
        raise RuntimeError(f'Missing posterior channels: {missing}')

    epochs = epochs[['cue_onset_right', 'cue_onset_left']].copy().pick(posterior_channels)

    # Combined attention-right and attention-left trials, as requested.
    tfr_raw = epochs.compute_tfr(
        method='multitaper',
        freqs=freqs,
        n_cycles=n_cycles,
        time_bandwidth=2.0,
        use_fft=True,
        return_itc=False,
        average=True,
        decim=2,
        n_jobs=4,
    )
    tfrs_raw[label] = tfr_raw

    # Separate copy for display only.
    tfr_plot = tfr_raw.copy()
    tfr_plot.apply_baseline(baseline=baseline, mode='percent')
    tfrs_plot[label] = tfr_plot

    out = op.join(deriv_folder, bids_path.basename + f'_both_{label}_tfr.h5')
    tfr_raw.save(out, overwrite=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, ch in zip(axes, posterior_channels):
        tfr_plot.plot(picks=ch, tmin=-0.3, tmax=1.4, baseline=None,
                      mode=None, axes=ax, show=False, colorbar=True)
        ax.set_title(f'{label}: {ch}')

    report.add_figure(
        fig,
        op.join(fig_folder, f'A02_{label}_three_channel_TFR.png'),
        f'{label}: combined attention-left/right TFR',
        f'Three posterior channels only; percent baseline {baseline}.',
        'Time-frequency analysis'
    )

# Stim minus no-stim for each channel separately, using raw TFR data.
difference = tfrs_raw['stim'].copy()
difference.data = tfrs_raw['stim'].data - tfrs_raw['no-stim'].data

fig_diff, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
for ax, ch in zip(axes, posterior_channels):
    difference.plot(picks=ch, tmin=-0.3, tmax=1.4, baseline=None,
                    mode=None, axes=ax, show=False, colorbar=True)
    ax.set_title(f'Stim - no-stim: {ch}')

report.add_figure(
    fig_diff,
    op.join(fig_folder, 'A02_stim_minus_no_stim_TFR.png'),
    'TFR difference: stimulation minus no stimulation',
    'Difference computed from unbaselined TFR data and shown separately for each posterior channel.',
    'Time-frequency analysis'
)

# Ratio: (stim on - stim off) / (stim on + stim off), using raw TFR data.
ratio = tfrs_raw['stim'].copy()
denom = tfrs_raw['stim'].data + tfrs_raw['no-stim'].data
eps = np.finfo(float).eps
ratio.data = (tfrs_raw['stim'].data - tfrs_raw['no-stim'].data) / (denom + eps)

fig_ratio, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
for ax, ch in zip(axes, posterior_channels):
    ratio.plot(picks=ch, tmin=-0.3, tmax=1.4, baseline=None,
               mode=None, axes=ax, show=False, colorbar=True)
    ax.set_title(f'Ratio: {ch}')

report.add_figure(
    fig_ratio,
    op.join(fig_folder, 'A02_stim_ratio_no_stim_TFR.png'),
    'TFR ratio: (stimulation - no stimulation) / (stimulation + no stimulation)',
    'Ratio computed from unbaselined TFR data and shown separately for each posterior channel.',
    'Time-frequency analysis'
)

subject_notes = input(                                                  # to add any notes to the PDF report for this subject, e.g. about data quality, artifacts, etc.
    f"\nFinal notes for sub-{subject} (press Enter to skip): "
).strip()

if subject_notes:
    report.add_text(
        "Subject notes",
        subject_notes,
        "Quality control",
    )

print(f'Updated PDF: {report.pdf_fname}')