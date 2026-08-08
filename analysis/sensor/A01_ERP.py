# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
===============================================
A01_ERP
    1. this code reads the cleaned cue epochs
    2. combines attention right and attention left
    epochs together
    3. makes evoked responses for stim on and stim off
    separately
    4. plots the comparison between stim on and stim
    off in one figure with different colours
    5. plots evoked responses for the posterior
    channels separately
    6. saves the evoked files and adds all figures to
    the participant PDF report

    note that this script keeps the comparison focused
    on the cue-locked epochs that survived manual
    cleaning in the previous step.

written by Tara Ghafari
tara.ghafari@gmail.com
==============================================

"""

import os
import os.path as op
import sys
import matplotlib.pyplot as plt
import mne
from mne_bids import BIDSPath

GITHUB_ROOT = r'/Users/taraghafari/Desktop/Desktop - Tara’s MacBook Pro/BEAR_outage/GitHub/STN-minimal-pdf-edits-v6'
UTILS_DIR = os.path.join(GITHUB_ROOT, 'analysis', 'utils')

if GITHUB_ROOT not in sys.path:
    sys.path.insert(0, GITHUB_ROOT)
if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)

from pdf_report import ParticipantPDF, impedance_text

subject = '115'
session = '01'
task = 'SpAtt'
run = '01'
eeg_suffix = 'eeg'
project_root = '/Users/taraghafari/Desktop/Desktop - Tara’s MacBook Pro/BEAR_outage/STN-in-PD'  # local folder
bids_root = op.join(project_root, 'data', 'BIDS')
posterior_channels = ['PO3', 'PO4', 'POz']

bids_path = BIDSPath(subject=subject, session=session, task=task, run=run,
                     root=bids_root, datatype='eeg', suffix=eeg_suffix)
deriv_folder = op.join(bids_root, 'derivatives', 'sub-' + subject)
fig_folder = op.join(project_root, 'derivatives', 'figures', f'sub-{subject}')
report_folder = op.join(project_root, 'derivatives', 'reports', f'sub-{subject}')
os.makedirs(fig_folder, exist_ok=True)
report = ParticipantPDF(report_folder, subject)

evokeds = {'cue': {}, 'grating': {}}

def make_evoked(epochs, tmin, tmax, baseline, shift=None):
    evoked = epochs.average(method='mean').filter(l_freq=None, h_freq=30)
    evoked = evoked.copy().crop(tmin=tmin, tmax=tmax)
    evoked.apply_baseline(baseline)
    if shift is not None:
        evoked = evoked.copy().shift_time(shift, relative=True)
    return evoked


def add_compare_fig(evoked_dict, fname, title, caption, picks, xlim):
    fig = mne.viz.plot_compare_evokeds(
        evoked_dict,
        picks=picks,
        combine='mean',
        show=False,
        ci=False,
        truncate_xaxis=False,
        truncate_yaxis=False,
    )
    if isinstance(fig, list):
        fig = fig[0]
    fig.axes[0].axvline(0, color='k', linestyle='--', linewidth=1)
    fig.axes[0].set_xlim(*xlim)
    report.add_figure(fig, fname, title, caption, 'Evoked responses')


for label in ['no-stim', 'stim']:
    input_fname = op.join(deriv_folder, bids_path.basename + f'_{label}_epo-cue.fif')
    epochs = mne.read_epochs(input_fname, preload=True)
    epochs = epochs[['cue_onset_right', 'cue_onset_left']]

    evokeds['cue'][label] = make_evoked(
        epochs, tmin=-0.1, tmax=0.5, baseline=(-0.1, 0)
    )
    evokeds['grating'][label] = make_evoked(
        epochs, tmin=1.1, tmax=1.6, baseline=(1.1, 1.2), shift=-1.2
    )

    mne.write_evokeds(
        op.join(deriv_folder, bids_path.basename + f'_{label}_evo-cue.fif'),
        evokeds['cue'][label],
        overwrite=True
    )
    mne.write_evokeds(
        op.join(deriv_folder, bids_path.basename + f'_{label}_evo-grating.fif'),
        evokeds['grating'][label],
        overwrite=True
    )

# Cue comparison, averaged across the 3 posterior channels
add_compare_fig(
    {'no stimulation': evokeds['cue']['no-stim'], 'stimulation': evokeds['cue']['stim']},
    op.join(fig_folder, 'A01_stim_no_stim_evoked_cue_comparison.png'),
    'Cue-locked evoked comparison: stimulation vs no stimulation',
    'Cue onset at 0 s; window -0.1 to 0.5 s; baseline -0.1 to 0 s; three posterior channels averaged.',
    posterior_channels,
    (-0.1, 0.5)
)

# Grating comparison, averaged across the 3 posterior channels
add_compare_fig(
    {'no stimulation': evokeds['grating']['no-stim'], 'stimulation': evokeds['grating']['stim']},
    op.join(fig_folder, 'A01_stim_no_stim_evoked_grating_comparison.png'),
    'Grating-locked evoked comparison: stimulation vs no stimulation',
    'Grating onset at 0 s; original window 1.1 to 1.6 s shifted by -1.2 s; baseline 1.1 to 1.2 s; three posterior channels averaged.',
    posterior_channels,
    (-0.1, 0.4)
)

# Cue comparison by channel
fig_cue_channels, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
for ax, ch in zip(axes, posterior_channels):
    mne.viz.plot_compare_evokeds(
        {'no stimulation': evokeds['cue']['no-stim'], 'stimulation': evokeds['cue']['stim']},
        picks=ch,
        combine=None,
        axes=ax,
        show=False,
        ci=False,
        truncate_xaxis=False,
        truncate_yaxis=False
    )
    ax.set_title(f'Cue-locked: {ch}')
    ax.axvline(0, color='k', linestyle='--', linewidth=1)
    ax.set_xlim(-0.1, 0.5)

report.add_figure(
    fig_cue_channels,
    op.join(fig_folder, 'A01_stim_no_stim_evoked_cue_by_channel.png'),
    'Cue-locked evoked responses by posterior channel',
    'Cue onset at 0 s; window -0.1 to 0.5 s; baseline -0.1 to 0 s; stimulation and no stimulation compared separately for each channel.',
    'Evoked responses'
)

# Grating comparison by channel
fig_grating_channels, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
for ax, ch in zip(axes, posterior_channels):
    mne.viz.plot_compare_evokeds(
        {'no stimulation': evokeds['grating']['no-stim'], 'stimulation': evokeds['grating']['stim']},
        picks=ch,
        combine=None,
        axes=ax,
        show=False,
        ci=False,
        truncate_xaxis=False,
        truncate_yaxis=False
    )
    ax.set_title(f'Grating-locked: {ch}')
    ax.axvline(0, color='k', linestyle='--', linewidth=1)
    ax.set_xlim(-0.1, 0.4)

report.add_figure(
    fig_grating_channels,
    op.join(fig_folder, 'A01_stim_no_stim_evoked_grating_by_channel.png'),
    'Grating-locked evoked responses by posterior channel',
    'Grating onset at 0 s; original window 1.1 to 1.6 s shifted by -1.2 s; baseline 1.1 to 1.2 s; stimulation and no stimulation compared separately for each channel.',
    'Evoked responses'
)
print(f'Updated PDF: {report.pdf_fname}')
