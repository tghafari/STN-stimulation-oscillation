#!/usr/bin/env python3
"""
===================================================================
Prepare reusable posterior alpha power and verified block assignments.

IMPORTANT: Run ALL intended participants together in ONE G04 invocation.
The output CSV files (trial_alpha.csv, block_summary.csv and mapping_audit.csv)
are replaced on every run. For testing a subset, supply --output-dir
with a separate directory so the full-group outputs are not overwritten. Running only one participant will overwrite
the previous multi-participant outputs; G04 does not append across runs.

Confirm each stimulation sequence before processing. Confirmed sequences
are saved immediately in analysis/subject/stimulation_sequence_confirmed.json.
Errors can be logged and participants skipped without excluding them from
later statistical analysis. Absolute trial-level alpha is not baselined.

Run from the repository root with all intended subjects:
python analysis/subject/EEG_posterior_channels/block_preparation/B03_prepare_block_alpha.py \\
    --subjects 101 102 103 105 106 107 108 112 113 114 115 116 \\
    118 120 121 123
===================================================================
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import mne
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parents[2]
SUBJECT_DIR = ANALYSIS / "subject"
for folder in (
    SUBJECT_DIR / "EEG_all_channels",
    HERE,
):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from pipeline_config import resolve_project_root, stage_path
from B01_review_bids_blocks import events_path, get_blocks, review_rows

from B02_validate_epoch_mapping import map_condition_epochs

# Edit these constants for subsequent analyses.
ATTENTION_WINDOW = (0.3, 0.8)
ALPHA_PEAK_RANGE = 1.0
FREQS = np.arange(2.0, 32.0, 0.5)
ROI8 = ("PO3", "POz", "PO4", "O1", "Oz", "O2", "PO7", "PO8")
CROP_FILE = SUBJECT_DIR / "stimulation_cropped_time.json"
CONFIRMED_FILE = SUBJECT_DIR / "stimulation_sequence_confirmed.json"


def parse_args():
    """Read subjects and file locations; Mac is the default platform."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--platform", choices=("mac", "bluebear"), default="mac")
    parser.add_argument("--project-root")
    parser.add_argument("--session", default="01")
    parser.add_argument("--task", default="SpAtt")
    parser.add_argument("--run", default="01")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--peaks-csv", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Write test runs to a separate directory to protect group CSVs.",
    )
    parser.add_argument(
        "--reconfirm", action="store_true",
        help="Ask again for previously confirmed stimulation sequences.",
    )
    return parser.parse_args()


def ask_on_error(subject, stage, exc, failures):
    """Record an error and ask whether to skip or stop.

    Never silently omit a participant. Preserve the full traceback in
    the error log for debugging, including errors in block confirmation.
    """
    detail = traceback.format_exc()
    failures.append({
        "subject": subject,
        "stage": stage,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": detail,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    })
    print(f"\nERROR sub-{subject} ({stage}): {exc}")
    while True:
        choice = input(
            "Skip this participant and continue [s], or stop [q]? "
        ).strip().lower()
        if choice in ("s", "skip"):
            return "skip"
        if choice in ("q", "quit"):
            return "quit"
        print("Please enter s or q.")


def choose_order(subject, rows):
    """Confirm or correct four stimulation periods directly within G04.

    This is the single interactive stimulation-order confirmation workflow.
    Confirmed values are then persisted by confirm_sequences().
    """
    proposed = rows["wiki_proposed_condition"].astype(str).str.lower().tolist()
    correct = sorted(("off", "off", "stim_left", "stim_right"))
    pretty = {
        "off": "STN OFF",
        "stim_left": "LEFT STN ON",
        "stim_right": "RIGHT STN ON",
    }

    def display(labels):
        print(f"\\nsub-{subject} - stimulation period sequence")
        print("Task-block pairs: (1,2), (3,4), (5,6), (7,8)")
        for (_, row), label in zip(rows.iterrows(), labels):
            print(
                f"  Task blocks {row.get('source_task_blocks', '?')}: "
                f"{pretty.get(label, label)}; "
                f"BIDS {row['onset_s']}-{row['offset_s']} s; "
                f"cue L={row['cue_left_n']}, R={row['cue_right_n']}"
            )

    if sorted(proposed) == correct:
        display(proposed)
        while True:
            answer = input(
                "Confirm order? [y/n/s(skip)/q(quit)]: "
            ).strip().lower()
            if answer == "y":
                return "confirmed", proposed
            if answer in ("s", "q"):
                return answer, None
            if answer == "n":
                break
    else:
        print(f"sub-{subject}: no valid saved proposal; enter manually.")

    print("Enter four chronological labels: o=OFF, l=left, r=right.")
    conversion = {"o": "off", "l": "stim_left", "r": "stim_right"}
    while True:
        raw = input("Four labels [l/r/o], or s(skip), q(quit): ")
        answer = raw.strip().lower()
        if answer in ("s", "q"):
            return answer, None
        tokens = answer.replace(",", " ").split()
        if len(tokens) == 4 and all(token in conversion for token in tokens):
            labels = [conversion[token] for token in tokens]
            if sorted(labels) == correct:
                display(labels)
                confirm = input("Confirm this order? [y/n]: ").strip().lower()
                if confirm == "y":
                    return "confirmed", labels
        print("Expected two OFF, one left ON and one right ON.")


def confirm_sequences(root, subjects, args, failures):
    """Confirm block orders using the existing l/r/o interface.

    Saved valid orders are reused without prompting unless --reconfirm is set.\n    Any failure prompts the user to skip or quit. A skipped participant
    remains eligible for future runs and is never excluded automatically.
    """
    if CONFIRMED_FILE.exists():
        raw_text = CONFIRMED_FILE.read_text(encoding="utf-8")
        # Earlier versions wrote a literal backslash-n after the JSON.
        raw_text = raw_text.strip()
        if raw_text.endswith(chr(92) + "n"):
            raw_text = raw_text[:-2].rstrip()
        confirmed = json.loads(raw_text)
    else:
        confirmed = {}
    for subject in subjects:
        saved = confirmed.get(f"sub-{subject}")
        if (
            not args.reconfirm
            and isinstance(saved, list)
            and sorted(saved) == sorted(
                ("off", "off", "stim_left", "stim_right")
            )
        ):
            print(f"sub-{subject}: using saved confirmed stimulation order.")
            continue
        try:
            rows = pd.DataFrame(
                review_rows(root, [subject], args.session, args.task, args.run)
            )
            if len(rows) != 4:
                raise ValueError(
                    f"Expected four stimulation periods, found {len(rows)}. "
                    f"Review note: {rows.iloc[0].get('review_note', 'unknown')}"
                )
            rows = rows.sort_values(
                "block_index_chronological"
            ).reset_index(drop=True)
            if f"sub-{subject}" in confirmed:
                rows["wiki_proposed_condition"] = confirmed[f"sub-{subject}"]
            action, labels = choose_order(subject, rows)
            if action == "q":
                raise KeyboardInterrupt("User quit during confirmation")
            if action == "s":
                failures.append({
                    "subject": subject, "stage": "confirmation",
                    "error_type": "UserSkip",
                    "error_message": "Participant skipped by user",
                    "traceback": "",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                })
                continue
            if action != "confirmed":
                raise ValueError("Stimulation sequence was not confirmed")
            confirmed[f"sub-{subject}"] = labels
            # Persist immediately, even if later participants fail.
            temporary = CONFIRMED_FILE.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(confirmed, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(CONFIRMED_FILE)
            print(f"Saved confirmed order: {CONFIRMED_FILE}")
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            if ask_on_error(subject, "confirmation", exc, failures) == "quit":
                break
    return confirmed


def crop_ranges(values):
    """Convert two/four crop endpoints into one/two time intervals."""
    numbers = [float(value) for value in values]
    if len(numbers) not in (2, 4):
        raise ValueError(f"Expected 2 or 4 crop endpoints, got {numbers}")
    return list(zip(numbers[::2], numbers[1::2]))


def load_epochs(root, subject, condition, args):
    """Read cleaned cue epochs, retaining only valid cue directions."""
    path = stage_path(
        root, subject, args.session, args.task, args.run,
        condition, "clean", "epo"
    )
    epochs = mne.read_epochs(path, preload=True, verbose=False)
    cues = [
        name for name in ("cue_onset_left", "cue_onset_right")
        if name in epochs.event_id
    ]
    if not cues:
        raise ValueError(f"sub-{subject}: no cue events in {path}")
    return epochs[cues]


def absolute_alpha(epochs, channels, peak, n_jobs):
    """Average linear multitaper power over channels, Hz and attention time.

    Return one value per trial; NO baseline correction or log transform.
    """
    tfr = epochs.copy().pick(channels).compute_tfr(
        method="multitaper",
        freqs=FREQS,
        n_cycles=FREQS / 2.0,
        time_bandwidth=2.0,
        use_fft=True,
        zero_mean=True,
        return_itc=False,
        average=False,
        decim=2,
        n_jobs=n_jobs,
        verbose=False,
    )
    frequency_mask = (
        (tfr.freqs >= peak - ALPHA_PEAK_RANGE - 1e-9)
        & (tfr.freqs <= peak + ALPHA_PEAK_RANGE + 1e-9)
    )
    time_mask = (
        (tfr.times >= ATTENTION_WINDOW[0])
        & (tfr.times <= ATTENTION_WINDOW[1])
    )
    if not frequency_mask.any() or not time_mask.any():
        raise ValueError("Requested frequency or attention window is empty")
    return tfr.data[:, :, frequency_mask, :][
        :, :, :, time_mask
    ].mean(axis=(1, 2, 3))


def prepare_subject(root, subject, peak, sequence, crops, args):
    """Build validated trial rows and audit records for one participant."""
    labels = sequence.get(f"sub-{subject}")
    if not labels or len(labels) != 4:
        raise ValueError(f"sub-{subject}: no four-period sequence in JSON")
    if sorted(labels) != sorted(("off", "off", "stim_left", "stim_right")):
        raise ValueError(f"sub-{subject}: invalid proposed labels {labels}")

    bids = pd.read_csv(
        events_path(root, subject, args.session, args.task, args.run),
        sep="\t",
    )
    blocks = get_blocks(bids)
    if len(blocks) != 8:
        raise ValueError(
            f"sub-{subject}: expected eight BIDS task blocks, got {len(blocks)}"
        )
    intervals = [
        (float(block["onset_s"]), float(block["offset_s"]))
        for block in blocks
    ]
    key = f"sub-{subject}"
    if key not in crops:
        raise ValueError(f"sub-{subject}: missing crop-time record")

    epochs = {
        condition: load_epochs(root, subject, condition, args)
        for condition in ("no-stim", "stim")
    }
    channels = [
        channel for channel in ROI8
        if all(
            channel in ep.ch_names and channel not in ep.info["bads"]
            for ep in epochs.values()
        )
    ]
    if not channels:
        raise ValueError(f"sub-{subject}: no shared good posterior channels")

    trial_rows = []
    audits = []
    for condition, ep in epochs.items():
        diagnostic_root = (
            args.output_dir
            if args.output_dir is not None
            else root / "derivatives" / "reports" / "group"
            / "EEG_posterior_channels_block_mapping"
        )
        diagnostic_path = (
            diagnostic_root / "diagnostics"
            / f"sub-{subject}_{condition}_timing.csv"
        )
        mapping = map_condition_epochs(
            ep,
            bids,
            crop_ranges(crops[key][condition]),
            intervals,
            condition,
            diagnostic_csv=diagnostic_path,
        )
        values = absolute_alpha(ep, channels, peak, args.n_jobs)
        indices = mapping["matched_task_block_indices"]
        on_label = condition == "stim"
        for index, (task_index, power) in enumerate(zip(indices, values)):
            period = int(task_index) // 2
            stimulation = labels[period]
            if (stimulation != "off") != on_label:
                raise ValueError(
                    f"sub-{subject}: confirmed condition disagrees with "
                    f"cleaned epochs in period {period + 1}"
                )
            event_code = int(ep.events[index, 2])
            cue = next(
                name for name, code in ep.event_id.items()
                if code == event_code
            )
            trial_rows.append({
                "subject": subject,
                "trial_index_in_condition": index,
                "source_condition": condition,
                "task_block": int(task_index) + 1,
                "stimulation_period": period + 1,
                "stimulation_side": stimulation,
                "dbs_on": stimulation != "off",
                "cue_direction": "left" if cue.endswith("left") else "right",
                "bids_cue_onset_s": mapping["matched_bids_onsets_s"][index],
                "block_start_s": intervals[task_index][0],
                "block_end_s": intervals[task_index][1],
                "block_onset_source": blocks[task_index]["onset_source"],
                "block_onset_inferred": blocks[task_index]["onset_inferred"],
                "alpha_peak_hz": peak,
                "alpha_power": float(power),
                "n_posterior_channels": len(channels),
                "channels": ";".join(channels),
                "label_source": "interactive_user_confirmation",
                "label_verified": True,
            })
        audits.append({
            "subject": subject,
            "source_condition": condition,
            "clean_epochs": len(ep),
            "mapped_epochs": len(indices),
            "sample_offset": mapping["sample_offset"],
            "segment_sample_offsets": str(
                mapping.get("segment_sample_offsets", [])
            ),
            "segment_split_epoch": mapping.get("segment_split_epoch"),
            "coordinate_system": mapping["coordinate_system"],
            "unretained_bids_cues": mapping["n_unretained_bids_cues"],
            "inferred_task_blocks": ";".join(
                str(i + 1) for i, block in enumerate(blocks)
                if block["onset_inferred"]
            ),
            "label_source": "interactive_user_confirmation",
            "label_verified": True,
        })
    if len(trial_rows) != sum(map(len, epochs.values())):
        raise ValueError(f"sub-{subject}: incomplete trial mapping")
    return trial_rows, audits


def save_outputs(output, trials, audits, failures):
    """Write partial successful data and complete diagnostics to disk."""
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        failures,
        columns=[
            "subject", "stage", "error_type", "error_message",
            "traceback", "timestamp_utc",
        ],
    ).to_csv(output / "G04_error_log.csv", index=False)
    if not trials:
        print("No successful participant data to save.")
        return
    frame = pd.DataFrame(trials)
    frame.to_csv(output / "trial_alpha.csv", index=False)
    summary = frame.groupby(
        ["subject", "stimulation_period", "task_block",
         "stimulation_side", "dbs_on"],
        as_index=False,
    ).agg(
        n_trials=("alpha_power", "size"),
        mean_absolute_alpha=("alpha_power", "mean"),
        block_start_s=("block_start_s", "first"),
        block_end_s=("block_end_s", "first"),
        block_onset_source=("block_onset_source", "first"),
        block_onset_inferred=("block_onset_inferred", "first"),
    )
    summary.to_csv(output / "block_summary.csv", index=False)
    pd.DataFrame(audits).to_csv(output / "mapping_audit.csv", index=False)


def main():
    """Process confirmed participants, offering skip/stop on failures."""
    args = parse_args()
    subjects = [value.removeprefix("sub-") for value in args.subjects]
    if len(subjects) != len(set(subjects)):
        raise ValueError("Duplicate subject identifiers supplied")
    root = resolve_project_root(args.platform, args.project_root)
    output = (
        root / "derivatives" / "reports" / "group"
        / "EEG_posterior_channels_block_mapping"
    )
    if args.output_dir is not None:
        output = args.output_dir.expanduser().resolve()
    failures = []
    trials = []
    audits = []

    try:
        # Confirmation is completed before any EEG power calculation.
        sequence = confirm_sequences(root, subjects, args, failures)
        peaks_path = args.peaks_csv or (
            root / "derivatives" / "reports" / "group"
            / "individual_alpha_peaks.csv"
        )
        peaks = pd.read_csv(peaks_path, dtype={"subject": str})
        peaks["subject"] = peaks["subject"].str.removeprefix("sub-")
        if peaks["subject"].duplicated().any():
            raise ValueError("Duplicate subject in alpha-peak CSV")
        peaks = peaks.set_index("subject")
        crops = json.loads(CROP_FILE.read_text(encoding="utf-8"))

        for subject in subjects:
            if f"sub-{subject}" not in sequence:
                continue
            try:
                if subject not in peaks.index:
                    raise ValueError("Missing individual alpha peak")
                peak = float(peaks.loc[subject, "peak_hz"])
                if not np.isfinite(peak):
                    raise ValueError("Invalid individual alpha peak")
                trial_rows, audit_rows = prepare_subject(
                    root, subject, peak, sequence, crops, args
                )
                trials.extend(trial_rows)
                audits.extend(audit_rows)
                print(
                    f"sub-{subject}: mapped {len(trial_rows)} cleaned trials"
                )
            except Exception as exc:
                if ask_on_error(subject, "processing", exc, failures) == "quit":
                    break
    finally:
        save_outputs(output, trials, audits, failures)
        print(f"Saved trial-level input and QC PDF in: {output}")
        print(f"Successful participants: {len(set(row['subject'] for row in trials))}")
        print(f"Skipped/error records: {len(failures)}")




if __name__ == "__main__":
    main()
