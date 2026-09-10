"""A03: add a concise manuscript-style analysis overview to the participant PDF."""
from __future__ import annotations

import argparse

from pipeline_config import resolve_project_root
from all_channel_report import participant_report


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subject', required=True)
    p.add_argument('--platform', choices=['mac', 'bluebear'], default='mac')
    p.add_argument('--project-root', default=None)
    return p.parse_args()


def main():
    a = parse_args()
    subject = a.subject.removeprefix('sub-')
    root = resolve_project_root(a.platform, a.project_root)
    report = participant_report(root, subject)

    text = (
        'EEG data were low-pass filtered at 100 Hz and noisy channels were identified with PyPREP on the full continuous recording, '
        'followed by manual channel-quality review. Stimulation and no-stimulation periods were then segmented using the predefined '
        'stimulation timing table. Cue-locked epochs from -0.5 to +1.6 s were created and average-referenced using the retained EEG channels. '
        'A single FastICA decomposition was fitted to the concatenated rereferenced stimulation and no-stimulation epochs using a 1-40 Hz, '
        '200-Hz fitting copy; only clearly artifactual components were removed. Remaining contaminated trials were rejected manually. '
        'Final spectral quality control used Welch power spectra from 0.1 to 100 Hz with an FFT length of up to 2 s, and stimulation-related '
        'spectral contamination was characterized by comparing stimulation with no-stimulation power. Cue-locked ERPs were averaged across '
        'retained trials, low-pass filtered at 30 Hz and baseline corrected from -0.1 to 0 s. Time-frequency power was estimated from 2 to 30 Hz '
        'with multitaper convolution using frequency/2 cycles and a time-bandwidth product of 2. Condition-specific TFR displays were percent '
        'baseline corrected from -0.3 to -0.1 s. In contrast, stimulation-minus-no-stimulation power and the normalized difference '
        '(stimulation - no stimulation)/(stimulation + no stimulation) were calculated from unbaselined power so that baseline normalization '
        'did not influence the direct condition comparisons. Analyses were performed across all available common good EEG sensors and were '
        'also visualized separately for PO3, PO4 and POz when those posterior electrodes were available.'
    )

    report.add_text(
        'Participant EEG analysis: methods summary',
        text,
        'Analysis overview (manuscript style)',
    )
    print(f'Analysis overview added for sub-{subject}: {report.pdf_fname}')


if __name__ == '__main__':
    main()
