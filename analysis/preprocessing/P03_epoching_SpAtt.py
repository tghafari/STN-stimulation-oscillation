# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
===============================================
05_epoching_SpAtt
    1. this code reads the stim on and stim off
    segmented files
    2. filters the data between 0.1 and 100 Hz
    3. epochs the cue onsets from -0.5 to 1.6 sec
    4. computes the PSD of the epochs
    5. finds bad channels using pyprep and
    writes the reasons into the PDF report
    6. opens a plot that shows only three posterior
    channels ('PO3', 'PO4', 'POz') so the user can manually 
    reject bad trials
    7. saves the cleaned epochs for later analysis

    note that the manual rejection should be based
    only on the three posterior channels chosen for
    this step, while the rejected trial is removed
    from all channels in the epoch.

written by Tara Ghafari
tara.ghafari@gmail.com
==============================================

"""

import json
import os
import os.path as op
import sys
from copy import deepcopy
import matplotlib.pyplot as plt
import mne
import numpy as np
from mne_bids import BIDSPath

GITHUB_ROOT = r'/Users/taraghafari/Desktop/Desktop - Tara’s MacBook Pro/BEAR_outage/GitHub/STN-minimal-pdf-edits-v6'
UTILS_DIR = os.path.join(GITHUB_ROOT, 'analysis', 'utils')

if GITHUB_ROOT not in sys.path:
    sys.path.insert(0, GITHUB_ROOT)
if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)

from pdf_report import ParticipantPDF

# PyPREP is used only to suggest noisy channels and reasons.
from pyprep.find_noisy_channels import NoisyChannels

subject = '115'
session = '01'
task = 'SpAtt'
run = '01'
eeg_suffix = 'eeg'
extension = '.fif'
project_root = '/Users/taraghafari/Desktop/Desktop - Tara’s MacBook Pro/BEAR_outage/STN-in-PD'  # local folder
bids_root = op.join(project_root, 'data', 'BIDS')
posterior_channels = ['PO3', 'PO4', 'POz']  # These are the ones we agreed with Ole

event_dict = {'cue_onset_right': 1, 'cue_onset_left': 2, 'trial_onset': 3,
              'stim_onset': 4, 'catch_onset': 5, 'dot_onset_right': 6,
              'dot_onset_left': 7, 'response_press_onset': 8,
              'block_onset': 20, 'block_end': 21, 'experiment_end': 30,
              'new_stim_segment': 99999}

bids_path = BIDSPath(subject=subject, session=session, task=task, run=run,
                     root=bids_root, datatype='eeg', suffix=eeg_suffix)
deriv_folder = op.join(bids_root, 'derivatives', 'sub-' + subject)
fig_folder = op.join(project_root, 'derivatives', 'figures', f'sub-{subject}')
report_folder = op.join(project_root, 'derivatives', 'reports', f'sub-{subject}')
os.makedirs(fig_folder, exist_ok=True)
report = ParticipantPDF(report_folder, subject)


def get_bad_channel_reasons(raw):
    """Run PyPREP detectors and return channel -> reason list."""
    eeg = raw.copy().pick('eeg')
    if eeg.get_montage() is None:
        eeg.set_montage('standard_1020', on_missing='warn')
    noisy = NoisyChannels(eeg, random_state=42)
    detectors = [
        ('deviation', noisy.find_bad_by_deviation),
        ('high-frequency noise', noisy.find_bad_by_hfnoise),
        ('correlation', noisy.find_bad_by_correlation),
        ('RANSAC', noisy.find_bad_by_ransac),  # Random sample consensus, or RANSAC works by identifying the outliers 
                                               # in a data set and estimating the desired model using data that does not 
                                               # contain outliers
    ]
    reasons = {}
    for reason, detector in detectors:
        try:
            detector()
        except Exception as exc:
            print(f'PyPREP {reason} detector skipped: {exc}')
    # PyPREP stores lists under these attributes after each detector.
    attr_map = {
        'bad_by_deviation': 'deviation',
        'bad_by_hf_noise': 'high-frequency noise',
        'bad_by_correlation': 'correlation',
        'bad_by_ransac': 'RANSAC',
        'bad_by_nan': 'NaN/flat data',
        'bad_by_SNR': 'poor signal-to-noise ratio',
    }
    for attr, reason in attr_map.items():
        for ch in getattr(noisy, attr, []) or []:
            reasons.setdefault(ch, []).append(reason)
    for ch in noisy.get_bads():
        reasons.setdefault(ch, []).append('PyPREP overall noisy-channel decision')
    return {ch: sorted(set(vals)) for ch, vals in reasons.items()}

def format_channel_list(channels):
    channels = [str(ch) for ch in channels]
    channels = sorted(set(channels))
    return ", ".join(channels) if channels else "None"

segment_data = {}
all_bad_channels = set()

for label in ['no-stim', 'stim']:
    input_fname = op.join(deriv_folder, bids_path.basename + f'_{label}_raw.fif')
    raw = mne.io.read_raw_fif(input_fname, preload=True)

    reasons = get_bad_channel_reasons(raw)
    suggested = sorted(reasons)

    print(f'PyPREP suggested bad channels for {label}: {suggested}')
    print(json.dumps(reasons, indent=2))

    raw.compute_psd(fmin=0.1, fmax=150).plot()  # to look at all channels and remove obvious bad ones
    user = input(
        'Additional bad channels, separated by spaces, or press return: '
    ).strip().split()
    manual_reason = input(
        'Optional manual reason for these additional channels: '
    ).strip()

    for ch in user:
        reasons.setdefault(str(ch), []).append(
            manual_reason or 'manually identified during QC'
        )

    # collect bad channels from this segment
    segment_bad_channels = set(str(ch) for ch in raw.info['bads'])
    segment_bad_channels.update(str(ch) for ch in reasons.keys())
    all_bad_channels.update(segment_bad_channels)

    segment_data[label] = {
        'raw': raw,
        'reasons': reasons,
    }

# make the bad-channel set common to both segments
common_bads = sorted(all_bad_channels)

for label in ['no-stim', 'stim']:
    raw = segment_data[label]['raw']
    reasons = segment_data[label]['reasons']

    bads_to_remove = [ch for ch in common_bads if ch in raw.ch_names]

    # remove the same bad channels from both segments
    raw.info['bads'] = bads_to_remove
    raw.drop_channels(bads_to_remove)

    bad_text = "\n".join(sorted(set(str(ch) for ch in bads_to_remove))) or "None"

    reason_text = "\n".join(
    f"{str(ch)}: {', '.join(map(str, reason_list))}"
    for ch, reason_list in sorted(reasons.items())
    ) or "No additional noisy channels detected."

    report.add_text(
        f'{label}: bad-channel reasons',
        f'Epochs -0.5 to 1.6 s; bad channels: {bad_text}\nReasons: {reason_text}',
        'Epoching and channel quality'
    )

    events, events_id = mne.events_from_annotations(raw, event_id=event_dict)
    cue_id = {
        k: events_id[k]
        for k in ['cue_onset_right', 'cue_onset_left']
        if k in events_id
    }

    epochs = mne.Epochs(
        raw,
        events,
        cue_id,
        tmin=-0.5,
        tmax=1.6,
        baseline=None,
        detrend=1,
        proj=True,
        picks='all',
        reject=None,
        reject_by_annotation=False,
        preload=True,
        event_repeated='merge',
    )

    n_fft = min(int(2 * epochs.info['sfreq']), len(epochs.times))
    fig_psd = epochs.compute_psd(
        fmin=0.1,
        fmax=100,
        method='welch',
        n_fft=n_fft
    ).plot(show=False)

    report.add_figure(
        fig_psd,
        op.join(fig_folder, f'P05_{label}_epoch_PSD.png'),
        f'{label}: PSD of cue epochs',
        f'Epochs -0.5 to 1.6 s; bad channels: {bad_text}',
        'Epoching and channel quality'
    )

    missing = [ch for ch in posterior_channels if ch not in epochs.ch_names]
    if missing:
        raise RuntimeError(f'Posterior QC channels missing: {missing}')

    n_before = len(epochs)
    epochs.plot(
        picks=posterior_channels,
        n_channels=3,
        block=True,
        title=f'{label}: manually reject trials using only {posterior_channels}'
    )
    n_after = len(epochs)

    output_fname = op.join(deriv_folder, bids_path.basename + f'_{label}_epo-cue.fif')
    epochs.save(output_fname, overwrite=True)

    report.add_text(
        f'{label}: manual posterior trial rejection',
        f'Channels shown: {posterior_channels}\n'
        f'Epochs before: {n_before}\n'
        f'Epochs retained: {n_after}\n'
        f'Epochs rejected: {n_before - n_after}\n'
        f'Output: {output_fname}',
        'Epoching and channel quality'
    )

print(f'Updated PDF: {report.pdf_fname}')
