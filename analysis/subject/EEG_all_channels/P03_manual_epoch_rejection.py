"""Stage 3: manual bad-trial rejection using all retained EEG channels.

ICA-cleaned epochs are inspected in MNE using all good EEG electrodes. Marked bad
channels remain in info['bads'] and are excluded from the browser. After manual
rejection, the final epoch PSD is computed with the same explicit Welch settings
used in the posterior-channel epoch PSD: fmin=0.1 Hz, fmax=100 Hz and an n_fft of
up to 2 seconds of data (limited by the epoch length). No butterfly ERP is made.
"""
from __future__ import annotations

import argparse
import json
import mne

from pipeline_config import CONDITIONS, qc_dir, resolve_project_root, stage_path
from all_channel_report import participant_report, figure_dir, fmt_channels


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subject', required=True)
    p.add_argument('--session', default='01')
    p.add_argument('--task', default='SpAtt')
    p.add_argument('--run', default='01')
    p.add_argument('--platform', choices=['mac', 'bluebear'], default='mac')
    p.add_argument('--project-root', default=None)
    p.add_argument('--n-channels', type=int, default=20)
    return p.parse_args()


def main():
    a = parse_args()
    subject = a.subject.removeprefix('sub-')
    root = resolve_project_root(a.platform, a.project_root)
    report = participant_report(root, subject)
    figs = figure_dir(root, subject)
    audit = {'subject': f'sub-{subject}', 'conditions': {}}

    report.add_text(
        'P03 method: manual bad-trial rejection',
        'ICA-cleaned cue epochs are visually inspected using all good EEG channels. '
        'Previously marked bad electrodes are not displayed. The researcher scrolls '
        'through the MNE epoch browser and manually marks contaminated trials. No '
        'amplitude threshold or automatic epoch rejection is applied at this stage. '
        'Bad channels remain marked rather than interpolated or dropped.',
        'All-channel EEG preprocessing',
    )

    for condition in CONDITIONS:
        infile = stage_path(root, subject, a.session, a.task, a.run, condition, 'ica', 'epo')
        if not infile.exists():
            raise FileNotFoundError(f'Missing ICA-cleaned epochs: {infile}')

        epochs = mne.read_epochs(infile, preload=True)
        n_before = len(epochs)
        bad_channels = list(epochs.info['bads'])
        good_eeg = [
            ch for ch in epochs.copy().pick('eeg').ch_names
            if ch not in bad_channels
        ]
        if not good_eeg:
            raise RuntimeError('No good EEG channels remain.')

        print('\n' + '=' * 72)
        print(f'sub-{subject} / {condition}: MANUAL BAD-TRIAL REJECTION')
        print(f'Epochs before: {n_before}')
        print(f'Bad channels excluded from display: {bad_channels or "None"}')
        print('=' * 72)

        epochs.plot(
            picks=good_eeg,
            n_channels=min(a.n_channels, len(good_eeg)),
            block=True,
            title=f'sub-{subject} {condition}: manual rejection using all good EEG channels',
        )

        n_after = len(epochs)
        outfile = stage_path(root, subject, a.session, a.task, a.run, condition, 'clean', 'epo')
        epochs.save(outfile, overwrite=True)

        # Match the explicit Welch PSD method used by EEG_posterior_channels/P03.
        # The previous all-channel P03 left method/n_fft at MNE defaults. With short
        # epochs that can give a visibly different spectral estimate and make narrow
        # line-noise features look misleadingly abrupt. Using the same settings makes
        # this final QC directly comparable with the earlier epoch PSDs.
        n_fft = min(int(2 * epochs.info['sfreq']), len(epochs.times))
        spectrum = epochs.compute_psd(
            fmin=0.1,
            fmax=min(100.0, epochs.info['sfreq'] / 2.0),
            method='welch',
            n_fft=n_fft,
        )
        fig_psd = spectrum.plot(show=False)
        report.add_figure(
            fig_psd,
            str(figs / f'P03_{condition}_final_epoch_PSD.png'),
            f'{condition}: PSD of final cleaned epochs',
            f'Welch PSD using the same settings as the posterior-channel epoch PSD. '
            f'Epochs retained: {n_after}; manually rejected: {n_before - n_after}.',
            'All-channel EEG preprocessing',
        )

        report.add_text(
            f'{condition}: manual trial-rejection result',
            f'Epochs before manual inspection: {n_before}\n'
            f'Epochs after manual inspection: {n_after}\n'
            f'Epochs manually rejected: {n_before - n_after}\n'
            f'Bad channels excluded from visual trial QC: {fmt_channels(bad_channels)}\n'
            f'Number of good EEG channels inspected: {len(good_eeg)}',
            'All-channel EEG preprocessing',
        )

        audit['conditions'][condition] = {
            'input_epochs': str(infile),
            'final_epochs': str(outfile),
            'n_epochs_before_manual_rejection': n_before,
            'n_epochs_after_manual_rejection': n_after,
            'n_epochs_manually_rejected': n_before - n_after,
            'bad_channels_retained_in_info': list(epochs.info['bads']),
            'good_eeg_channels_used_for_visual_trial_qc': good_eeg,
            'final_psd': {
                'method': 'welch',
                'fmin_hz': 0.1,
                'fmax_hz': min(100.0, epochs.info['sfreq'] / 2.0),
                'n_fft': n_fft,
            },
        }

    audit_file = qc_dir(root, subject) / 'P03_manual_epoch_rejection.json'
    audit_file.write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
    report.add_text(
        'Preprocessing complete',
        'All-channel preprocessing completed through manual bad-trial rejection. '
        'The participant PDF contains the inclusion decision, PyPREP/channel QC, '
        'rereferencing, ICA decisions, and final trial-rejection/PSD QC.',
        'All-channel EEG preprocessing',
    )
    print(f'\nUpdated PDF: {report.pdf_fname}')


if __name__ == '__main__':
    main()
