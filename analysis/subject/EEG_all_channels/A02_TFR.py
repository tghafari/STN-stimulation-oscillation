"""A02: all-channel and posterior-channel time-frequency analysis.

The final manually cleaned cue epochs from P03 are used. TFRs are computed with
multitaper power from 2-30 Hz in 1-Hz steps, n_cycles=freq/2,
time_bandwidth=2.0, FFT enabled, decimation=2, average=True, and no ITC.

Condition-specific displays (no-stim and stim) use percent baseline correction
from -0.3 to -0.1 s. Comparative TFRs (stim - no-stim and
(stim - no-stim)/(stim + no-stim)) are computed from the original unbaselined
power and are NOT baseline corrected.
"""
from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import mne
import numpy as np

from pipeline_config import CONDITIONS, qc_dir, resolve_project_root, stage_path, subject_deriv_dir
from all_channel_report import participant_report, figure_dir, fmt_channels

POSTERIOR = ("PO3", "PO4", "POz")
BASELINE = (-0.3, -0.1)
FREQS = np.arange(2.0, 31.0, 1.0)
N_CYCLES = FREQS / 2.0
TIME_BANDWIDTH = 2.0
DECIM = 2
PLOT_TMIN = -0.3
PLOT_TMAX = 1.4


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subject', required=True)
    p.add_argument('--session', default='01')
    p.add_argument('--task', default='SpAtt')
    p.add_argument('--run', default='01')
    p.add_argument('--platform', choices=['mac', 'bluebear'], default='mac')
    p.add_argument('--project-root', default=None)
    p.add_argument('--n-jobs', type=int, default=4)
    return p.parse_args()


def load_clean_epochs(root, subject, args):
    epochs = {}
    for condition in CONDITIONS:
        path = stage_path(root, subject, args.session, args.task, args.run, condition, 'clean', 'epo')
        if not path.exists():
            raise FileNotFoundError(f'Missing final cleaned epochs for {condition}: {path}')
        ep = mne.read_epochs(path, preload=True)
        if {'cue_onset_right', 'cue_onset_left'}.intersection(ep.event_id):
            keep = [k for k in ('cue_onset_right', 'cue_onset_left') if k in ep.event_id]
            ep = ep[keep]
        epochs[condition] = ep
    return epochs


def common_good_eeg(epochs):
    stim = epochs['stim']
    nostim = epochs['no-stim']
    stim_eeg = stim.copy().pick('eeg').ch_names
    return [
        ch for ch in stim_eeg
        if ch in nostim.ch_names
        and ch not in stim.info['bads']
        and ch not in nostim.info['bads']
    ]


def compute_tfr(ep, n_jobs):
    return ep.compute_tfr(
        method='multitaper',
        freqs=FREQS,
        n_cycles=N_CYCLES,
        time_bandwidth=TIME_BANDWIDTH,
        use_fft=True,
        return_itc=False,
        average=True,
        decim=DECIM,
        n_jobs=n_jobs,
    )


def posterior_figure(tfr, channels, title_prefix):
    fig, axes = plt.subplots(1, len(channels), figsize=(5 * len(channels), 4), constrained_layout=True)
    axes = [axes] if len(channels) == 1 else list(axes)
    for ax, ch in zip(axes, channels):
        tfr.plot(
            picks=ch,
            tmin=PLOT_TMIN,
            tmax=PLOT_TMAX,
            baseline=None,
            mode=None,
            axes=ax,
            show=False,
            colorbar=True,
        )
        ax.set_title(f'{title_prefix}: {ch}')
    return fig


def add_topo(report, tfr, image_path, title, caption, section='Time-frequency analysis'):
    fig = tfr.plot_topo(
        tmin=PLOT_TMIN,
        tmax=PLOT_TMAX,
        baseline=None,
        mode=None,
        show=False,
    )
    report.add_figure(fig, str(image_path), title, caption, section)


def main():
    a = parse_args()
    subject = a.subject.removeprefix('sub-')
    root = resolve_project_root(a.platform, a.project_root)
    report = participant_report(root, subject)
    figs = figure_dir(root, subject)
    deriv = subject_deriv_dir(root, subject)

    epochs = load_clean_epochs(root, subject, a)
    common = common_good_eeg(epochs)
    if not common:
        raise RuntimeError('No common good EEG sensors are available for TFR comparison.')

    posterior = [ch for ch in POSTERIOR if ch in common]

    # Compute raw/unbaselined TFRs on the exact same common good sensors so that
    # subtraction and normalized difference are channel-by-channel comparable.
    raw_tfr = {}
    plot_tfr = {}
    for condition in CONDITIONS:
        ep = epochs[condition].copy().pick(common)
        raw_tfr[condition] = compute_tfr(ep, a.n_jobs)

        # Save unbaselined power for later group/comparative analyses.
        out = deriv / f'sub-{subject}_ses-{a.session}_task-{a.task}_run-{a.run}_{condition}_desc-allchannels_tfr.h5'
        raw_tfr[condition].save(out, overwrite=True)

        # Condition-specific visualization: percent baseline only on a copy.
        plot_tfr[condition] = raw_tfr[condition].copy()
        plot_tfr[condition].apply_baseline(BASELINE, mode='percent')

        add_topo(
            report,
            plot_tfr[condition],
            figs / f'A02_{condition}_TFR_topo.png',
            f'{condition}: TFR topographic sensor layout',
            f'Multitaper TFR, 2-30 Hz in 1-Hz steps; n_cycles=f/2; time_bandwidth={TIME_BANDWIDTH:g}; '
            f'decim={DECIM}; percent baseline {BASELINE}; display window {PLOT_TMIN} to {PLOT_TMAX} s. '
            f'All {len(common)} common good EEG sensors are shown in sensor-layout form.',
        )

        if posterior:
            fig_post = posterior_figure(plot_tfr[condition], posterior, condition)
            report.add_figure(
                fig_post,
                str(figs / f'A02_{condition}_TFR_posterior.png'),
                f'{condition}: posterior-channel TFR',
                f'Posterior channels shown separately: {fmt_channels(posterior)}. Multitaper 2-30 Hz; '
                f'percent baseline {BASELINE}; display window {PLOT_TMIN} to {PLOT_TMAX} s.',
                'Time-frequency analysis',
            )

    # Comparative maps are deliberately derived from UNBASELINED power.
    difference = raw_tfr['stim'].copy()
    difference.data = raw_tfr['stim'].data - raw_tfr['no-stim'].data
    difference.save(
        deriv / f'sub-{subject}_ses-{a.session}_task-{a.task}_run-{a.run}_stim-minus-no-stim_desc-allchannels_tfr.h5',
        overwrite=True,
    )

    add_topo(
        report,
        difference,
        figs / 'A02_stim_minus_no_stim_TFR_topo.png',
        'TFR difference topographic sensor layout: stimulation - no stimulation',
        'Computed directly from unbaselined multitaper power. No baseline correction is applied to this comparative TFR. '
        f'Frequencies 2-30 Hz; n_cycles=f/2; time_bandwidth={TIME_BANDWIDTH:g}; decim={DECIM}; '
        f'display window {PLOT_TMIN} to {PLOT_TMAX} s.',
    )
    if posterior:
        fig_diff_post = posterior_figure(difference, posterior, 'stim - no-stim')
        report.add_figure(
            fig_diff_post,
            str(figs / 'A02_stim_minus_no_stim_TFR_posterior.png'),
            'TFR difference: posterior channels',
            f'Stimulation minus no stimulation from unbaselined power; no baseline correction. Posterior channels: {fmt_channels(posterior)}.',
            'Time-frequency analysis',
        )

    ratio = raw_tfr['stim'].copy()
    denom = raw_tfr['stim'].data + raw_tfr['no-stim'].data
    eps = np.finfo(float).eps
    ratio.data = (raw_tfr['stim'].data - raw_tfr['no-stim'].data) / (denom + eps)
    ratio.save(
        deriv / f'sub-{subject}_ses-{a.session}_task-{a.task}_run-{a.run}_stim-normalized-difference_desc-allchannels_tfr.h5',
        overwrite=True,
    )

    add_topo(
        report,
        ratio,
        figs / 'A02_stim_normalized_difference_TFR_topo.png',
        'TFR normalized difference topographic sensor layout: (stim - no-stim) / (stim + no-stim)',
        'Computed directly from unbaselined multitaper power. No baseline correction is applied. '
        f'Frequencies 2-30 Hz; n_cycles=f/2; time_bandwidth={TIME_BANDWIDTH:g}; decim={DECIM}; '
        f'display window {PLOT_TMIN} to {PLOT_TMAX} s.',
    )
    if posterior:
        fig_ratio_post = posterior_figure(ratio, posterior, '(stim-no-stim)/(stim+no-stim)')
        report.add_figure(
            fig_ratio_post,
            str(figs / 'A02_stim_normalized_difference_TFR_posterior.png'),
            'TFR normalized difference: posterior channels',
            f'(Stim - no-stim)/(stim + no-stim), computed from unbaselined power; no baseline correction. '
            f'Posterior channels: {fmt_channels(posterior)}.',
            'Time-frequency analysis',
        )

    if not posterior:
        report.add_text(
            'Posterior TFR unavailable',
            'None of PO3, PO4 or POz was available as a good sensor in both conditions. All-sensor TFR analyses were still completed.',
            'Time-frequency analysis',
        )

    details = {
        'subject': f'sub-{subject}',
        'input': 'P03 final manually cleaned cue epochs',
        'epoch_original_window_s': [-0.5, 1.6],
        'conditions': list(CONDITIONS),
        'attention_conditions_combined': ['cue_onset_right', 'cue_onset_left'],
        'method': 'multitaper',
        'frequencies_hz': FREQS.tolist(),
        'frequency_step_hz': 1.0,
        'n_cycles': 'frequency / 2',
        'time_bandwidth': TIME_BANDWIDTH,
        'use_fft': True,
        'return_itc': False,
        'average': True,
        'decim': DECIM,
        'condition_plot_baseline_s': list(BASELINE),
        'condition_plot_baseline_mode': 'percent',
        'comparison_baseline_correction': None,
        'comparison_definitions': {
            'difference': 'stim - no-stim',
            'normalized_difference': '(stim - no-stim) / (stim + no-stim)',
        },
        'common_good_eeg_sensors': common,
        'posterior_requested': list(POSTERIOR),
        'posterior_available': posterior,
        'n_epochs': {condition: len(epochs[condition]) for condition in CONDITIONS},
    }
    (qc_dir(root, subject) / 'A02_tfr_analysis.json').write_text(json.dumps(details, indent=2) + '\n', encoding='utf-8')

    report.add_text(
        'TFR analysis details',
        f'Input: final P03 cleaned cue epochs (-0.5 to +1.6 s), with attention-left and attention-right trials combined within each stimulation condition.\n'
        f'Method: multitaper power; 2-30 Hz in 1-Hz steps; n_cycles = frequency/2; time-bandwidth = {TIME_BANDWIDTH:g}; FFT=True; ITC=False; average=True; decimation={DECIM}.\n'
        f'Condition-specific no-stim and stim figures: percent baseline correction from {BASELINE[0]:g} to {BASELINE[1]:g} s.\n'
        f'Comparative stim-no-stim figures: computed from unbaselined power with NO baseline correction.\n'
        f'Difference: stim - no-stim.\n'
        f'Normalized difference: (stim - no-stim)/(stim + no-stim).\n'
        f'Display window: {PLOT_TMIN:g} to {PLOT_TMAX:g} s.\n'
        f'Common good EEG sensors used: {len(common)}. Posterior sensors available: {fmt_channels(posterior)}.',
        'Time-frequency analysis',
    )

    print(f'TFR complete for sub-{subject}. Updated PDF: {report.pdf_fname}')


if __name__ == '__main__':
    main()
