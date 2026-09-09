"""Stage 1: epoch definition, PyPREP bad-channel detection, and rereferencing.

Order and rationale
-------------------
1. Read the existing stim/no-stim continuous FIF derivatives created from BIDS.
2. Define cue-locked epochs (-0.5 to 1.6 s) BEFORE later cleaning decisions.
3. Run PyPREP on the corresponding continuous, pre-rereference EEG segment.
   PyPREP is designed for continuous data; therefore it is not run on an Epochs
   object. The detected channels are then transferred to the already-defined epochs.
4. Show PSDs and allow the researcher to add/remove bad-channel decisions.
5. Use the UNION of bad channels from stim and no-stim so both conditions are
   treated identically.
6. Apply average reference while excluding marked bad channels.
7. Save rereferenced continuous data, rereferenced epochs, and a transparent JSON
   audit trail. Bad channels are MARKED, not interpolated or physically dropped.

This preserves missing-channel information for later channel-wise group averages.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mne
from pyprep.find_noisy_channels import NoisyChannels

from pipeline_config import CONDITIONS, qc_dir, resolve_project_root, segmented_raw_path, stage_path

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject", required=True)
    p.add_argument("--session", default="01")
    p.add_argument("--task", default="SpAtt")
    p.add_argument("--run", default="01")
    p.add_argument("--platform", choices=["mac", "bluebear"], default="mac")
    p.add_argument("--project-root", default=None)
    p.add_argument("--line-freq", type=float, default=50.0)
    return p.parse_args()


def pyprep_reasons(raw: mne.io.BaseRaw) -> dict[str, list[str]]:
    """Return every PyPREP suggestion together with the detector(s) that found it."""
    eeg = raw.copy().pick("eeg")
    if eeg.get_montage() is None:
        eeg.set_montage("standard_1020", on_missing="warn")
    noisy = NoisyChannels(eeg, random_state=42)
    detectors = [
        ("deviation", noisy.find_bad_by_deviation),
        ("high-frequency noise", noisy.find_bad_by_hfnoise),
        ("correlation", noisy.find_bad_by_correlation),
        ("RANSAC", noisy.find_bad_by_ransac),
    ]
    reasons: dict[str, list[str]] = {}
    detector_errors: list[str] = []
    for label, func in detectors:
        try:
            func()
        except Exception as exc:
            detector_errors.append(f"{label}: {type(exc).__name__}: {exc}")

    attr_map = {
        "bad_by_deviation": "deviation",
        "bad_by_hf_noise": "high-frequency noise",
        "bad_by_correlation": "correlation",
        "bad_by_ransac": "RANSAC",
        "bad_by_nan": "NaN/flat data",
        "bad_by_SNR": "poor signal-to-noise ratio",
    }
    for attr, why in attr_map.items():
        for ch in getattr(noisy, attr, []) or []:
            reasons.setdefault(str(ch), []).append(why)
    for ch in noisy.get_bads():
        reasons.setdefault(str(ch), []).append("PyPREP overall decision")
    if detector_errors:
        reasons["__detector_errors__"] = detector_errors
    return {ch: sorted(set(vals)) for ch, vals in reasons.items()}


def define_epochs(raw: mne.io.BaseRaw) -> mne.Epochs:
    """Create the same cue-locked epochs used by the posterior-channel pipeline."""
    events, event_ids = mne.events_from_annotations(raw, event_id=EVENT_DICT)
    cue_ids = {k: event_ids[k] for k in ("cue_onset_right", "cue_onset_left") if k in event_ids}
    if not cue_ids:
        raise RuntimeError("No cue_onset_right/left events found in this segment.")
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


def ask_manual_bad_edits(raw: mne.io.BaseRaw, suggested: list[str], condition: str) -> tuple[list[str], list[str]]:
    """Show PSD and collect explicit additions/removals from the researcher."""
    raw.info["bads"] = sorted(set(raw.info["bads"]) | set(suggested))
    raw.compute_psd(fmin=0.5, fmax=min(100.0, raw.info["sfreq"] / 2.0)).plot()
    print(f"\n{condition}: PyPREP suggests: {suggested or 'None'}")
    additions = input("Additional bad EEG channels (space-separated, Enter for none): ").strip().split()
    removals = input("PyPREP channels you believe are actually GOOD (space-separated, Enter for none): ").strip().split()
    return additions, removals


def main() -> None:
    args = parse_args()
    subject = args.subject.removeprefix("sub-")
    root = resolve_project_root(args.platform, args.project_root)

    raws: dict[str, mne.io.BaseRaw] = {}
    epochs_by_condition: dict[str, mne.Epochs] = {}
    audit: dict = {"subject": f"sub-{subject}", "conditions": {}, "common_bad_channels": []}
    union_bads: set[str] = set()

    for condition in CONDITIONS:
        infile = segmented_raw_path(root, subject, args.session, args.task, args.run, condition)
        if not infile.exists():
            raise FileNotFoundError(
                f"Missing input: {infile}\n"
                "This pipeline intentionally starts from the existing BIDS-derived stim/no-stim segments. "
                "Run the repository's stimulation-segmentation step first."
            )
        raw = mne.io.read_raw_fif(infile, preload=True, verbose=True)
        if raw.info.get("line_freq") is None:
            raw.info["line_freq"] = args.line_freq
        if raw.get_montage() is None:
            raw.set_montage("standard_1020", on_missing="warn")

        # Define epochs now, before channel-cleaning decisions.
        epochs = define_epochs(raw)

        reasons = pyprep_reasons(raw)
        detector_errors = reasons.pop("__detector_errors__", [])
        suggested = sorted(reasons)
        additions, removals = ask_manual_bad_edits(raw, suggested, condition)

        final_condition_bads = (set(raw.info["bads"]) | set(additions) | set(suggested)) - set(removals)
        final_condition_bads &= set(raw.ch_names)
        union_bads.update(final_condition_bads)

        audit["conditions"][condition] = {
            "input": str(infile),
            "n_epochs_defined": len(epochs),
            "pyprep_reasons": reasons,
            "pyprep_detector_errors": detector_errors,
            "manual_additions": additions,
            "manual_removals": removals,
            "condition_bad_channels": sorted(final_condition_bads),
        }
        raws[condition] = raw
        epochs_by_condition[condition] = epochs

    common_bads = sorted(union_bads)
    audit["common_bad_channels"] = common_bads

    print("\nCOMMON BAD CHANNELS USED FOR BOTH CONDITIONS:")
    print(common_bads or "None")
    print("These channels will remain marked bad; they are NOT interpolated or dropped.")

    for condition in CONDITIONS:
        raw = raws[condition]
        epochs = epochs_by_condition[condition]
        present_bads = [ch for ch in common_bads if ch in raw.ch_names]
        raw.info["bads"] = present_bads
        epochs.info["bads"] = [ch for ch in present_bads if ch in epochs.ch_names]

        # Average reference excludes channels in info['bads'].
        raw.set_eeg_reference(ref_channels="average", projection=False)
        epochs.set_eeg_reference(ref_channels="average", projection=False)

        raw_out = stage_path(root, subject, args.session, args.task, args.run, condition, "reref", "raw")
        epo_out = stage_path(root, subject, args.session, args.task, args.run, condition, "reref", "epo")
        raw.save(raw_out, overwrite=True)
        epochs.save(epo_out, overwrite=True)
        audit["conditions"][condition]["rereferenced_raw"] = str(raw_out)
        audit["conditions"][condition]["rereferenced_epochs"] = str(epo_out)

    audit_file = qc_dir(root, subject) / "P01_bad_channels_and_rereference.json"
    audit_file.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved QC audit: {audit_file}")


if __name__ == "__main__":
    main()
