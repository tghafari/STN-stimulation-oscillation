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
import warnings
import os
import os.path as op
import sys
from copy import deepcopy
import matplotlib.pyplot as plt
import mne
import numpy as np
from mne_bids import BIDSPath

GITHUB_ROOT = r'/Users/taraghafari/Desktop/Desktop - Tara’s MacBook Pro/BEAR_outage/GitHub/STN-stimulation-oscillation'
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
GROUP_POSTERIOR_CHANNELS = ['PO3', 'PO4', 'POz']

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

interpolation_summary = {
    "subject": subject,
    "interpolated_channels": [],
}

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


    # Mark the PyPREP bad channels before plotting the PSD.
    # This makes them visible as already-bad channels during inspection.
    raw.info["bads"] = sorted(set(raw.info["bads"]) | set(suggested))

    # Plot PSD with the PyPREP bad channels already flagged.
    raw.compute_psd(fmin=0.1, fmax=150).plot()  # to look at all channels and remove obvious bad ones
    user = input(
        'Additional bad channels, separated by spaces, or press return: '
    ).strip().split()

    # Prevent the user from re-entering channels PyPREP already found.
    user = [ch for ch in user if ch not in suggested]
    
    for ch in user:
        manual_reason = input(
            f"Reason for manually rejecting {ch}: "
        ).strip()

        reasons.setdefault(str(ch), []).append(
            manual_reason or "manually identified during QC"
        )

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
    # raw.drop_channels(bads_to_remove)  # we need to interpolate bad posterior channels for group analysis, so don't drop.

    bad_text = "\n".join(sorted(set(str(ch) for ch in bads_to_remove))) or "None"

    reason_text = "\n".join(
    f"{str(ch)}: {', '.join(map(str, reason_list))}"
    for ch, reason_list in sorted(reasons.items())
    ) or "No additional noisy channels detected."

    report.add_text(
        f'{label}: bad-channel reasons',
        f'Epochs -0.5 to 1.6 s; \nReasons: {reason_text}',
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
        f'Epochs -0.5 to 1.6 s, cue onset = 0s',
        'Epoching and channel quality'
    )

    posterior_channels = ['PO3', 'PO4', 'POz']

    # channels that are going to be removed and are also part of the downstream
    # posterior-channel analysis
    rejected_posterior = [ch for ch in posterior_channels if ch in common_bads]

    posterior_channels_for_analysis = posterior_channels.copy()

    if rejected_posterior:
        warnings.warn(
            f"The following rejected channels are part of the posterior-channel "
            f"analysis pipeline: {rejected_posterior}. "
            f"Do you want to continue using only the remaining posterior channels?",
            RuntimeWarning,
        )
        answer = input("Continue with remaining posterior channels only? [y/N]: ").strip().lower()

        if answer not in {"y", "yes"}:
            raise RuntimeError("Stopped because a posterior channel was rejected.")

        posterior_channels_for_analysis = [
            ch for ch in posterior_channels if ch not in rejected_posterior
        ]

        if not posterior_channels_for_analysis:
            raise RuntimeError("All posterior channels were rejected; cannot continue.")

        posterior_file = op.join(
            bids_root,
            "derivatives",
            f"sub-{subject}",
            "qc",
            f"sub-{subject}_posterior_channels.json",
        )
        os.makedirs(op.dirname(posterior_file), exist_ok=True)
        with open(posterior_file, "w", encoding="utf-8") as f:
            json.dump(posterior_channels_for_analysis, f, indent=2)

        report.add_text(
            "⚠️ WARNING: Posterior analysis channel rejected",
            f"""
        One or more posterior channels used for the downstream ERP and TFR analyses
        were rejected during EEG cleaning.

        Rejected posterior channel(s):
        {', '.join(rejected_posterior)}

        The user chose to continue the analysis.

        All subsequent analyses (manual epoch rejection, ERP and TFR) were performed
        using only the remaining posterior channel(s):

        {', '.join(posterior_channels_for_analysis)}

        Interpret the results with caution because the predefined posterior ROI
        was incomplete for this participant.
        """,
            "IMPORTANT WARNINGS",
        )

    # keep only the channels that remain
    n_before = len(epochs)
    epochs.plot(
        picks=posterior_channels_for_analysis,
        n_channels=len(posterior_channels_for_analysis),
        block=True,
        title=f"{label}: manually reject trials using only {posterior_channels_for_analysis}",
    )
    n_after = len(epochs)

    # ------------------------------------------------------------------
    # Prepare a group-analysis version of the epochs.
    #
    # Bad posterior channels are interpolated here so that all subjects
    # can contribute the same posterior ROI to the group analysis.
    # The ordinary subject-level epochs below remain cleaned with bad
    # channels removed.
    # ------------------------------------------------------------------

    group_epochs = epochs.copy()

    # Make sure the posterior channels have valid sensor positions.
    if group_epochs.get_montage() is None:
        group_epochs.set_montage(
            "standard_1020",
            on_missing="warn",
        )

    # Only interpolate posterior channels that were identified as bad.
    posterior_bads_to_interpolate = [
        ch
        for ch in rejected_posterior
        if ch in group_epochs.ch_names
    ]

    if posterior_bads_to_interpolate:

        group_epochs.info["bads"] = posterior_bads_to_interpolate

        group_epochs.interpolate_bads(
            reset_bads=True,
            mode="accurate",
        )

        print(
            f"{label}: interpolated posterior channel(s): "
            f"{posterior_bads_to_interpolate}"
        )

    interpolation_summary["interpolated_channels"].extend(
    posterior_bads_to_interpolate
    )

    output_fname = op.join(deriv_folder, bids_path.basename + f'_{label}_epo-cue.fif')
    epochs.save(output_fname, overwrite=True)

    group_output_fname = op.join(
    deriv_folder,
    bids_path.basename + f'_{label}_epo-cue-group.fif'
    )

    group_epochs.save(
        group_output_fname,
        overwrite=True,
    )

    report.add_text(
        f'{label}: manual posterior trial rejection',
        f'Channels shown: {posterior_channels}\n'
        f'Epochs before: {n_before}\n'
        f'Epochs retained: {n_after}\n'
        f'Epochs rejected: {n_before - n_after}\n'
        f'Output: {output_fname}',
        'Epoching and channel quality'
    )

interpolation_summary["interpolated_channels"] = sorted(
    set(interpolation_summary["interpolated_channels"])
)

interpolation_dir = op.join(
    bids_root,
    "derivatives",
    f"sub-{subject}",
    "qc",
)

os.makedirs(
    interpolation_dir,
    exist_ok=True,
)

interpolation_fname = op.join(
    interpolation_dir,
    f"sub-{subject}_group_interpolation.json",
)

with open(
    interpolation_fname,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        interpolation_summary,
        f,
        indent=2,
    )

print(f'Updated PDF: {report.pdf_fname}')
