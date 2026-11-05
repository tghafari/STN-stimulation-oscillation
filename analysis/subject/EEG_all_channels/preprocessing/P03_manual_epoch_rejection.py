"""Manually reject ICA-cleaned EEG epochs and document spectral QC.

Inputs are the two condition-specific ICA-cleaned epoch FIF files from P02.
The researcher inspects the epochs, manually rejects bad trials and can
mark additional EEG channels bad. The union of bad channels is applied to
both conditions; no interpolation is performed.

Outputs are final clean epoch FIF files, Welch PSD figures, stimulation
artifact QC and a JSON audit of retained epochs and channel decisions.
The stimulation PSD comparison is diagnostic only; it does not alter data.
"""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import mne
import numpy as np

from all_channel_report import figure_dir, fmt_channels, participant_report
from pipeline_config import (
    CONDITIONS,
    qc_dir,
    resolve_project_root,
    stage_path,
)


def parse_args():
    """Parse the participant and epoch-browser display settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--session", default="01")
    parser.add_argument("--task", default="SpAtt")
    parser.add_argument("--run", default="01")
    parser.add_argument(
        "--platform", choices=["mac", "bluebear"], default="mac"
    )
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--n-channels", type=int, default=20)
    return parser.parse_args()


def welch(epochs):
    """Compute Welch PSD with approximately two-second FFT segments."""
    n_fft = min(
        int(2 * epochs.info["sfreq"]),
        len(epochs.times),
    )
    spectrum = epochs.compute_psd(
        fmin=0.1,
        fmax=min(100.0, epochs.info["sfreq"] / 2.0),
        method="welch",
        n_fft=n_fft,
    )
    return spectrum, n_fft


def mean_psd(spectrum):
    """Average spectral power across epochs and channels."""
    return spectrum.get_data().mean(axis=(0, 1))


def stimulation_qc(clean, report, figures, subject):
    """Compare the two conditions' PSD using shared good EEG channels."""
    stim = clean["stim"]
    no_stim = clean["no-stim"]
    common = [
        channel for channel in stim.copy().pick("eeg").ch_names
        if channel in no_stim.ch_names
        and channel not in stim.info["bads"]
        and channel not in no_stim.info["bads"]
    ]

    stim_spectrum, _ = welch(stim.copy().pick(common))
    off_spectrum, _ = welch(no_stim.copy().pick(common))
    frequencies = stim_spectrum.freqs
    tiny = np.finfo(float).tiny
    stim_db = 10 * np.log10(
        np.maximum(mean_psd(stim_spectrum), tiny)
    )
    off_db = 10 * np.log10(
        np.maximum(mean_psd(off_spectrum), tiny)
    )

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(frequencies, off_db, label="no-stim")
    axis.plot(frequencies, stim_db, label="stim")
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Mean EEG PSD (dB)")
    axis.set_title(f"sub-{subject}: stimulation artifact QC")
    axis.legend()
    axis.grid(True, alpha=0.25)
    axis.set_xticks(
        np.arange(0, int(np.floor(frequencies.max() / 10)) * 10 + 1, 10)
    )
    axis.set_xlim(0, min(100.0, float(frequencies.max())))
    figure.tight_layout()
    report.add_figure(
        figure,
        str(figures / "P03_stimulation_artifact_QC.png"),
        "Stimulation artifact QC: stim versus no-stim PSD",
        (
            "Mean Welch PSD across epochs and the same "
            f"{len(common)} good EEG channels. Diagnostic only."
        ),
        "Stimulation artifact QC",
    )
    return {
        "common_good_eeg_channels": common,
        "n_common_good_eeg_channels": len(common),
        "qc_only_no_correction": True,
    }


def main():
    """Inspect and save final clean epochs and a detailed QC audit."""
    args = parse_args()
    subject = args.subject.removeprefix("sub-")
    root = resolve_project_root(args.platform, args.project_root)
    report = participant_report(root, subject)
    figures = figure_dir(root, subject)
    epochs = {}
    audit = {"subject": f"sub-{subject}", "conditions": {}}

    # Load both conditions before inspection. The interactive MNE
    # browser updates each Epochs object when bad trials are dropped.
    for condition in CONDITIONS:
        path = stage_path(
            root, subject, args.session, args.task, args.run,
            condition, "ica", "epo",
        )
        epochs[condition] = mne.read_epochs(path, preload=True)

    eeg = epochs["stim"].copy().pick("eeg").ch_names
    existing = sorted(
        set().union(
            *(set(epochs[condition].info["bads"]) for condition in CONDITIONS)
        )
    )
    before = {}
    for condition in CONDITIONS:
        before[condition] = len(epochs[condition])
        good = [
            channel for channel in epochs[condition].copy().pick("eeg").ch_names
            if channel not in epochs[condition].info["bads"]
        ]
        epochs[condition].plot(
            picks=good,
            n_channels=min(args.n_channels, len(good)),
            block=True,
            title=f"sub-{subject} {condition}: manual bad-trial QC",
        )

    # Ask for any additional bad channels after reviewing both conditions.
    while True:
        additional = input(
            "Additional EEG channels to mark BAD after epoch inspection "
            "(space/comma separated, Enter for none): "
        ).replace(",", " ").split()
        unknown = [channel for channel in additional if channel not in eeg]
        if not unknown:
            break
        print("Unknown EEG channels:", unknown)

    new = sorted(set(additional) - set(existing))
    reasons = {}
    for channel in new:
        while True:
            reason = input(f"Reason for marking {channel} bad: ").strip()
            if reason:
                break
            print("Please enter a reason.")
        reasons[channel] = reason

    final = sorted(set(existing) | set(additional))
    clean = {}
    for condition in CONDITIONS:
        epochs[condition].info["bads"] = [
            channel for channel in final
            if channel in epochs[condition].ch_names
        ]
        output = stage_path(
            root, subject, args.session, args.task, args.run,
            condition, "clean", "epo",
        )
        epochs[condition].save(output, overwrite=True)
        clean[condition] = epochs[condition].copy()

        spectrum, n_fft = welch(epochs[condition])
        figure = spectrum.plot(show=False)
        report.add_figure(
            figure,
            str(figures / f"P03_{condition}_final_epoch_PSD.png"),
            f"{condition}: PSD of final cleaned epochs",
            (
                f"Welch PSD; retained epochs {len(epochs[condition])}; "
                f"manually rejected {before[condition] - len(epochs[condition])}."
            ),
            "All-channel EEG preprocessing",
        )
        plt.close(figure)
        audit["conditions"][condition] = {
            "n_epochs_before_manual_rejection": before[condition],
            "n_epochs_after_manual_rejection": len(epochs[condition]),
            "final_bad_channels": final,
            "final_psd": {
                "method": "welch",
                "fmin_hz": 0.1,
                "fmax_hz": min(100.0, epochs[condition].info["sfreq"] / 2.0),
                "n_fft": n_fft,
            },
        }

    audit["additional_bad_channels_at_epoch_qc"] = new
    audit["additional_bad_channel_reasons"] = reasons
    audit["final_bad_channels"] = final
    audit["stimulation_artifact_qc"] = stimulation_qc(
        clean, report, figures, subject
    )
    output = qc_dir(root, subject) / "P03_manual_epoch_rejection.json"
    output.write_text(json.dumps(audit, indent=2) + "\n")

    reason_text = "; ".join(
        f"{channel}: {reason}" for channel, reason in reasons.items()
    ) or "None"
    report.add_text(
        "P03 manual epoch and channel QC",
        (
            "Bad trials were rejected manually. Additional bad channels: "
            f"{fmt_channels(new)}. Reasons: {reason_text}. "
            f"Final bad channels: {fmt_channels(final)}."
        ),
        "All-channel EEG preprocessing",
    )
    print(f"Updated PDF: {report.pdf_fname}")


if __name__ == "__main__":
    main()
