"""Filter, inspect, segment and epoch all-channel EEG for both DBS states.

Prerequisite: posterior-channel processing has produced BIDS EEG and
participant-specific stimulation crop intervals.

Steps
-----
1. Read continuous BIDS EEG and set a standard montage if necessary.
2. Apply the 1-100 Hz band-pass filter before running PyPREP.
3. Collect PyPREP noisy-channel suggestions, retaining detector errors.
4. Inspect raw channels manually; record additional bad channels/reasons.
5. Review rejected-channel PSD, optionally rescue channels.
6. Inspect the remaining good-channel PSD and mark further bad channels.
7. Crop continuous EEG into the two DBS conditions using the shared JSON.
8. Create cue epochs (-0.5 to +1.6 s, detrend=1, no baseline).
9. Optionally apply average EEG reference and save reference-labelled FIFs.
10. Record all QC decisions, crop times and epoch counts in JSON/report.

Bad channels are marked, never interpolated. Manual decisions are
interactive and must be recorded before proceeding to ICA.
"""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import mne
from mne_bids import read_raw_bids
from pyprep.find_noisy_channels import NoisyChannels

from all_channel_report import figure_dir, fmt_channels, participant_report
from pipeline_config import (
    CONDITIONS,
    base_bids_path,
    crop_table_path,
    qc_dir,
    resolve_project_root,
    stage_path,
)

EVENT_DICT = {
    "cue_onset_right": 1,
    "cue_onset_left": 2,
    "trial_onset": 3,
    "stim_onset": 4,
    "catch_onset": 5,
    "dot_onset_right": 6,
    "dot_onset_left": 7,
    "response_press_onset": 8,
    "block_onset": 20,
    "block_end": 21,
    "experiment_end": 30,
    "new_stim_segment": 99999,
}


def parse_args():
    """Read BIDS identifiers, platform and optional EEG reference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--session", default="01")
    parser.add_argument("--task", default="SpAtt")
    parser.add_argument("--run", default="01")
    parser.add_argument(
        "--platform", choices=["mac", "bluebear"], default="mac"
    )
    parser.add_argument("--project-root", default=None)
    parser.add_argument(
        "--rereference", choices=["avg", "none"], default="avg"
    )
    return parser.parse_args()


def pyprep_reasons(raw):
    """Collect per-channel PyPREP flags and nonfatal detector failures.

    Run each detector separately, preserving results even if another
    detector raises an exception. The manual review is authoritative.
    """
    eeg = raw.copy().pick("eeg")
    if eeg.get_montage() is None:
        eeg.set_montage("standard_1020", on_missing="warn")
    noisy = NoisyChannels(eeg, random_state=42)
    errors = []
    detectors = [
        ("deviation", noisy.find_bad_by_deviation),
        ("high-frequency noise", noisy.find_bad_by_hfnoise),
        ("correlation", noisy.find_bad_by_correlation),
        ("RANSAC", noisy.find_bad_by_ransac),
    ]
    for label, detector in detectors:
        try:
            detector()
        except Exception as exc:
            errors.append(
                f"{label}: {type(exc).__name__}: {exc}"
            )

    attributes = {
        "bad_by_deviation": "deviation",
        "bad_by_hf_noise": "high-frequency noise",
        "bad_by_correlation": "correlation",
        "bad_by_ransac": "RANSAC",
        "bad_by_nan": "NaN/flat data",
        "bad_by_SNR": "poor signal-to-noise ratio",
    }
    reasons = {}
    for attribute, reason in attributes.items():
        for channel in getattr(noisy, attribute, []) or []:
            reasons.setdefault(str(channel), []).append(reason)
    for channel in noisy.get_bads():
        reasons.setdefault(str(channel), []).append(
            "PyPREP overall decision"
        )
    return {
        key: sorted(set(values)) for key, values in reasons.items()
    }, errors


def make_segment(raw, times):
    """Crop one or two intervals and concatenate only when required."""
    pieces = [
        raw.copy().crop(
            tmin=float(times[0]), tmax=float(times[1])
        )
    ]
    if len(times) == 4:
        pieces.append(
            raw.copy().crop(
                tmin=float(times[2]), tmax=float(times[3])
            )
        )
    if len(pieces) == 1:
        return pieces[0]
    return mne.concatenate_raws(pieces, on_mismatch="warn")


def define_epochs(raw):
    """Create unbaselined, detrended cue-locked epochs from annotations."""
    events, identifiers = mne.events_from_annotations(
        raw, event_id=EVENT_DICT
    )
    cue_ids = {
        key: identifiers[key]
        for key in ("cue_onset_right", "cue_onset_left")
        if key in identifiers
    }
    if not cue_ids:
        raise RuntimeError(
            "No cue onset events found after stimulation segmentation."
        )
    return mne.Epochs(
        raw,
        events,
        cue_ids,
        tmin=-0.5,
        tmax=1.6,
        baseline=None,
        detrend=1,
        proj=True,
        picks="all",
        reject=None,
        reject_by_annotation=False,
        preload=True,
        event_repeated="merge",
    )


def ask_reason(channel, prompt):
    """Require a human-readable reason for a manual channel rejection."""
    while True:
        reason = input(prompt.format(channel=channel)).strip()
        if reason:
            return reason
        print("Please enter a reason.")


def review_rejected_psd(raw, rejected, report, figures):
    """Show rejected-channel spectra and allow explicit rescue."""
    rescued = []
    if rejected:
        rejected_raw = raw.copy().pick(rejected)
        rejected_raw.info["bads"] = []
        spectrum = rejected_raw.compute_psd(
            fmin=1.0,
            fmax=min(100.0, rejected_raw.info["sfreq"] / 2.0),
        )
        figure = spectrum.plot(show=False)
        report.add_figure(
            figure,
            str(figures / "P01_rejected_channels_PSD.png"),
            "PSD of PyPREP + manually rejected channels",
            (
                "1-100 Hz filtered data. Proposed rejection: "
                f"{fmt_channels(rejected)}."
            ),
            "All-channel preprocessing",
        )
        plt.close(figure)
        spectrum.plot(show=True)
        plt.show(block=True)
        rescued = input(
            "Rejected channels that are actually GOOD "
            "(space-separated, Enter for none): "
        ).strip().split()
    unknown = [channel for channel in rescued if channel not in rejected]
    if unknown:
        raise ValueError(
            f"Can only rescue channels shown in PSD: {unknown}"
        )
    return rescued


def good_eeg_channels(raw, bad_channels):
    """List EEG channels that are not marked bad."""
    return [
        channel for channel in raw.copy().pick("eeg").ch_names
        if channel not in bad_channels
    ]


def show_good_psd(raw, good, report, figures, filename, title, description):
    """Calculate and save a diagnostic PSD for the retained EEG channels."""
    selected = raw.copy().pick(good)
    selected.info["bads"] = []
    spectrum = selected.compute_psd(
        fmin=1.0, fmax=min(100.0, selected.info["sfreq"] / 2.0)
    )
    figure = spectrum.plot(show=False)
    report.add_figure(
        figure,
        str(figures / filename),
        title,
        description,
        "All-channel preprocessing",
    )
    plt.close(figure)
    return spectrum


def main():
    """Perform channel QC and save condition-specific cue epochs."""
    args = parse_args()
    subject = args.subject.removeprefix("sub-")
    root = resolve_project_root(args.platform, args.project_root)
    report = participant_report(root, subject)
    figures = figure_dir(root, subject)
    rereference = "avg" if args.rereference == "avg" else None
    reference_label = (
        "average EEG reference" if rereference else "none"
    )
    raw = read_raw_bids(
        base_bids_path(
            root, subject, args.session, args.task, args.run
        ),
        verbose=True,
        extra_params={"preload": True},
    )
    if raw.get_montage() is None:
        raw.set_montage("standard_1020", on_missing="warn")

    print("\nBand-pass filtering continuous EEG: 1-100 Hz before PyPREP.")
    raw.filter(l_freq=1.0, h_freq=100.0)
    reasons, detector_errors = pyprep_reasons(raw)
    suggested = sorted(reasons)
    raw.info["bads"] = sorted(
        set(raw.info["bads"]) | set(suggested)
    )
    print("PyPREP suggested:", suggested or "None")

    # Inspect the continuous recording and record manual additions.
    before = set(raw.info["bads"])
    raw.plot(
        n_channels=min(30, len(raw.ch_names)),
        duration=20.0,
        block=True,
        title=f"sub-{subject}: post-PyPREP manual channel QC",
    )
    after = set(raw.info["bads"])
    manual = sorted(after - before)
    browser_rescues = sorted(before - after)
    manual_reasons = {
        channel: ask_reason(
            channel,
            "Reason for manually rejecting {channel}: ",
        )
        for channel in manual
    }

    rejected = sorted(after)
    rescued = review_rejected_psd(
        raw, rejected, report, figures
    )
    final = sorted(set(rejected) - set(rescued))
    raw.info["bads"] = final
    good = good_eeg_channels(raw, final)
    if not good:
        raise RuntimeError("No good EEG channels remain.")

    spectrum = show_good_psd(
        raw, good, report, figures,
        "P01_final_good_channels_PSD.png",
        "PSD of remaining good EEG channels before final PSD review",
        (
            "1-100 Hz filtered retained EEG before segmentation: "
            f"{fmt_channels(good)}."
        ),
    )
    spectrum.plot(show=True)
    plt.show(block=True)

    while True:
        psd_bad = input(
            "Additional EEG channels to mark BAD based on the "
            "remaining-channel PSD (space/comma separated, "
            "Enter for none): "
        ).replace(",", " ").split()
        unknown = [channel for channel in psd_bad if channel not in good]
        if not unknown:
            break
        print("Can only reject currently good EEG channels:", unknown)

    psd_reasons = {
        channel: ask_reason(
            channel, "Reason for rejecting {channel} based on PSD: "
        )
        for channel in psd_bad
    }
    final = sorted(set(final) | set(psd_bad))
    raw.info["bads"] = final
    good = good_eeg_channels(raw, final)
    if not good:
        raise RuntimeError(
            "No good EEG channels remain after PSD review."
        )

    if psd_bad:
        show_good_psd(
            raw, good, report, figures,
            "P01_final_good_channels_PSD_after_PSD_rejection.png",
            "Final PSD after PSD-based channel rejection",
            (
                "Additional PSD-based rejection: "
                f"{fmt_channels(psd_bad)}. "
                f"Remaining EEG: {fmt_channels(good)}."
            ),
        )

    table = json.loads(crop_table_path().read_text())
    key = f"sub-{subject}"
    audit = {
        "subject": key,
        "high_pass_hz": 1.0,
        "low_pass_hz": 100.0,
        "filter": "1-100 Hz band-pass on continuous EEG before PyPREP",
        "rereference": rereference,
        "rereference_requested": args.rereference,
        "pyprep_reasons": reasons,
        "detector_errors": detector_errors,
        "pyprep_suggested_bad_channels": suggested,
        "manual_raw_browser_additions": manual,
        "manual_raw_browser_reasons": manual_reasons,
        "raw_browser_rescued_pyprep_channels": browser_rescues,
        "channels_rescued_after_rejected_psd": rescued,
        "additional_bad_channels_from_remaining_channel_psd": psd_bad,
        "additional_bad_channel_psd_reasons": psd_reasons,
        "bad_channels": final,
        "final_good_eeg_channels": good,
        "conditions": {},
    }

    # Apply the saved crop intervals and epoch each condition separately.
    for condition in CONDITIONS:
        times = table[key][condition]
        segment = make_segment(raw, times)
        segment.info["bads"] = [
            channel for channel in final
            if channel in segment.ch_names
        ]
        epochs = define_epochs(segment)
        epochs.info["bads"] = [
            channel for channel in final
            if channel in epochs.ch_names
        ]
        if rereference == "avg":
            epochs.set_eeg_reference(
                ref_channels="average", projection=False
            )
        output = stage_path(
            root, subject, args.session, args.task, args.run,
            condition, "reref", "epo",
        )
        epochs.save(output, overwrite=True)
        audit["conditions"][condition] = {
            "crop_times_sec": times,
            "n_epochs": len(epochs),
            "rereference": reference_label,
            "output_file": str(output),
        }
        report.add_text(
            f"{condition}: stimulation segmentation",
            (
                "Continuous EEG was already band-pass filtered 1-100 Hz.\n"
                f"Kept ranges (s): {times}\n"
                "Cue epochs: -0.5 to +1.6 s; baseline=None; detrend=1.\n"
                f"Rereferencing: {reference_label}.\n"
                f"Epochs: {len(epochs)}"
            ),
            "Stimulation segmentation",
        )

    psd_reason_text = "; ".join(
        f"{channel}: {reason}"
        for channel, reason in psd_reasons.items()
    ) or "None"
    report.add_text(
        "PyPREP, manual channel QC, PSD review, filtering and reference",
        (
            "Continuous EEG was band-pass filtered 1-100 Hz before PyPREP.\n"
            f"PyPREP suggestions: {fmt_channels(suggested)}\n"
            "User-rejected channels in raw browser: "
            f"{fmt_channels(manual)}\n"
            "Channels rescued after QC: "
            f"{fmt_channels(sorted(set(browser_rescues) | set(rescued)))}\n"
            "Additional channels rejected after remaining-channel "
            f"PSD review: {fmt_channels(psd_bad)}\n"
            f"PSD-based reasons: {psd_reason_text}\n"
            f"FINAL bad channels: {fmt_channels(final)}\n"
            f"Rereferencing: {reference_label}."
        ),
        "All-channel preprocessing",
    )
    (qc_dir(root, subject) / "P01_pyprep_segment_epoch_reref.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
