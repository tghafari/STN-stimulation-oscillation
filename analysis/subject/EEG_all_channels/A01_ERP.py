"""A01: ERP analysis for all available EEG sensors and posterior channels.

Reads the final manually cleaned stim and no-stim cue epochs from P03. ERP
comparisons are based on the common good EEG sensors present in both conditions.
A 30-Hz low-pass is applied to the evoked responses and a -0.1 to 0 s cue
baseline is used. One figure shows the stimulation/no-stimulation comparison
averaged across all common good EEG sensors; a second figure shows PO3, PO4 and
POz separately when available.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mne

from pipeline_config import CONDITIONS, qc_dir, resolve_project_root, stage_path
from all_channel_report import participant_report, figure_dir, fmt_channels

POSTERIOR = ("PO3", "PO4", "POz")
ERP_BASELINE = (-0.1, 0.0)
ERP_LP_HZ = 30.0
ERP_TMIN = -0.1
ERP_TMAX = 0.5


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subject', required=True)
    p.add_argument('--session', default='01')
    p.add_argument('--task', default='SpAtt')
    p.add_argument('--run', default='01')
    p.add_argument('--platform', choices=['mac', 'bluebear'], default='mac')
    p.add_argument('--project-root', default=None)
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


def make_evoked(epochs, picks):
    ep = epochs.copy().pick(picks)
    ev = ep.average(method='mean')
    ev.filter(l_freq=None, h_freq=ERP_LP_HZ)
    ev.apply_baseline(ERP_BASELINE)
    ev.crop(tmin=ERP_TMIN, tmax=ERP_TMAX)
    return ev


def main():
    a = parse_args()
    subject = a.subject.removeprefix('sub-')
    root = resolve_project_root(a.platform, a.project_root)
    report = participant_report(root, subject)
    figs = figure_dir(root, subject)

    epochs = load_clean_epochs(root, subject, a)
    common = common_good_eeg(epochs)
    if not common:
        raise RuntimeError('No common good EEG sensors are available for ERP comparison.')

    evoked = {condition: make_evoked(epochs[condition], common) for condition in CONDITIONS}

    # Save evoked data for later subject/group analysis.
    for condition in CONDITIONS:
        out = stage_path(root, subject, a.session, a.task, a.run, condition, 'erp', 'ave')
        mne.write_evokeds(out, evoked[condition], overwrite=True)

    compare = {'no stimulation': evoked['no-stim'], 'stimulation': evoked['stim']}

    # All available sensors: summarize the stimulation comparison as the mean
    # across the exact same common good EEG sensors in both conditions.
    fig_all = mne.viz.plot_compare_evokeds(
        compare,
        picks=common,
        combine='mean',
        show=False,
        ci=False,
        truncate_xaxis=False,
        truncate_yaxis=False,
    )
    if isinstance(fig_all, list):
        fig_all = fig_all[0]
    ax = fig_all.axes[0]
    ax.axvline(0, color='k', linestyle='--', linewidth=1)
    ax.set_xlim(ERP_TMIN, ERP_TMAX)
    ax.set_title(f'sub-{subject}: ERP, mean across {len(common)} common good EEG sensors')
    report.add_figure(
        fig_all,
        str(figs / 'A01_ERP_stim_vs_no_stim_all_sensors.png'),
        'ERP: stimulation vs no stimulation, all available sensors',
        f'Cue-locked ERP averaged across the same {len(common)} common good EEG sensors. '
        f'Evoked responses were low-pass filtered at {ERP_LP_HZ:g} Hz, baseline-corrected '
        f'from {ERP_BASELINE[0]:g} to {ERP_BASELINE[1]:g} s, and displayed from '
        f'{ERP_TMIN:g} to {ERP_TMAX:g} s. Cue onset = 0 s.',
        'ERP analysis',
    )

    posterior = [ch for ch in POSTERIOR if ch in common]
    if posterior:
        fig_post, axes = plt.subplots(1, len(posterior), figsize=(5 * len(posterior), 4), constrained_layout=True)
        axes = [axes] if len(posterior) == 1 else list(axes)
        for ax, ch in zip(axes, posterior):
            mne.viz.plot_compare_evokeds(
                compare,
                picks=ch,
                combine=None,
                axes=ax,
                show=False,
                ci=False,
                truncate_xaxis=False,
                truncate_yaxis=False,
            )
            ax.axvline(0, color='k', linestyle='--', linewidth=1)
            ax.set_xlim(ERP_TMIN, ERP_TMAX)
            ax.set_title(ch)
        report.add_figure(
            fig_post,
            str(figs / 'A01_ERP_stim_vs_no_stim_posterior.png'),
            'ERP: stimulation vs no stimulation, posterior channels',
            f'PO3/PO4/POz shown separately when available. Available posterior sensors: {fmt_channels(posterior)}. '
            f'30-Hz low-pass; baseline {ERP_BASELINE}; cue onset = 0 s; display window {ERP_TMIN} to {ERP_TMAX} s.',
            'ERP analysis',
        )
    else:
        report.add_text(
            'Posterior ERP unavailable',
            'None of PO3, PO4 or POz was available as a good sensor in both conditions.',
            'ERP analysis',
        )

    details = {
        'subject': f'sub-{subject}',
        'input': 'P03 final manually cleaned cue epochs',
        'epoch_original_window_s': [-0.5, 1.6],
        'erp_display_window_s': [ERP_TMIN, ERP_TMAX],
        'baseline_s': list(ERP_BASELINE),
        'evoked_low_pass_hz': ERP_LP_HZ,
        'averaging': 'arithmetic mean across retained trials',
        'all_sensor_plot': 'condition comparison after mean across common good EEG sensors',
        'common_good_eeg_sensors': common,
        'posterior_requested': list(POSTERIOR),
        'posterior_available': posterior,
        'n_epochs': {condition: len(epochs[condition]) for condition in CONDITIONS},
    }
    (qc_dir(root, subject) / 'A01_erp_analysis.json').write_text(json.dumps(details, indent=2) + '\n', encoding='utf-8')

    report.add_text(
        'ERP analysis details',
        f'Input: final P03 cleaned cue epochs (-0.5 to +1.6 s).\n'
        f'Conditions: stimulation and no stimulation.\n'
        f'Trials: attention-left and attention-right cue epochs combined within each condition.\n'
        f'Averaging: arithmetic mean across retained trials.\n'
        f'ERP low-pass: {ERP_LP_HZ:g} Hz.\n'
        f'Baseline correction: {ERP_BASELINE[0]:g} to {ERP_BASELINE[1]:g} s relative to cue onset.\n'
        f'Display window: {ERP_TMIN:g} to {ERP_TMAX:g} s.\n'
        f'Common good EEG sensors used: {len(common)}.\n'
        f'Posterior sensors available: {fmt_channels(posterior)}.',
        'ERP analysis',
    )

    print(f'ERP complete for sub-{subject}. Updated PDF: {report.pdf_fname}')


if __name__ == '__main__':
    main()
