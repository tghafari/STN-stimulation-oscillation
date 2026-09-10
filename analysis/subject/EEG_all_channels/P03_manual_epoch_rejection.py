"""Stage 3: manual bad-trial rejection using all retained EEG channels.

The ICA-cleaned epochs are opened in the standard MNE epoch browser. Mark bad
trials there and close the browser when finished. MNE removes the marked epochs
from the Epochs object. The script then saves final cleaned epochs plus a JSON
record of trial counts before/after rejection.

Important: channels previously identified as bad remain listed in info['bads'].
They are not interpolated and not physically dropped, so downstream group code
can exclude a participant separately for each bad electrode.
"""
from __future__ import annotations

import argparse
import json

import mne

from pipeline_config import CONDITIONS, qc_dir, resolve_project_root, stage_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject", required=True)
    p.add_argument("--session", default="01")
    p.add_argument("--task", default="SpAtt")
    p.add_argument("--run", default="01")
    p.add_argument("--platform", choices=["mac", "bluebear"], default="mac")
    p.add_argument("--project-root", default=None)
    p.add_argument(
        "--n-channels",
        type=int,
        default=20,
        help="Number of channels visible at once in the epoch browser (scroll to see the rest).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    subject = args.subject.removeprefix("sub-")
    root = resolve_project_root(args.platform, args.project_root)
    audit = {"subject": f"sub-{subject}", "conditions": {}}

    for condition in CONDITIONS:
        infile = stage_path(root, subject, args.session, args.task, args.run, condition, "ica", "epo")
        if not infile.exists():
            raise FileNotFoundError(f"Missing ICA-cleaned epochs: {infile}")
        epochs = mne.read_epochs(infile, preload=True)

        n_before = len(epochs)
        bad_channels = list(epochs.info["bads"])
        good_eeg = [ch for ch in epochs.copy().pick("eeg").ch_names if ch not in bad_channels]
        if not good_eeg:
            raise RuntimeError("No good EEG channels remain for manual trial inspection.")

        print("\n" + "=" * 72)
        print(f"sub-{subject} / {condition}: MANUAL BAD-TRIAL REJECTION")
        print(f"Epochs before inspection: {n_before}")
        print(f"Marked bad channels (excluded from display): {bad_channels or 'None'}")
        print("Inspect ALL good EEG channels by scrolling through the browser.")
        print("Click an epoch to mark/unmark it bad; close the browser when finished.")
        print("=" * 72)

        epochs.plot(
            picks=good_eeg,
            n_channels=min(args.n_channels, len(good_eeg)),
            block=True,
            title=f"sub-{subject} {condition}: manual rejection using all good EEG channels",
        )

        n_after = len(epochs)
        final_out = stage_path(root, subject, args.session, args.task, args.run, condition, "clean", "epo")
        epochs.save(final_out, overwrite=True)

        audit["conditions"][condition] = {
            "input_epochs": str(infile),
            "final_epochs": str(final_out),
            "n_epochs_before_manual_rejection": n_before,
            "n_epochs_after_manual_rejection": n_after,
            "n_epochs_manually_rejected": n_before - n_after,
            "bad_channels_retained_in_info": list(epochs.info["bads"]),
            "good_eeg_channels_used_for_visual_trial_qc": good_eeg,
        }
        print(f"Saved {n_after} cleaned epochs ({n_before - n_after} rejected): {final_out}")

    audit_file = qc_dir(root, subject) / "P03_manual_epoch_rejection.json"
    audit_file.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved manual-rejection audit: {audit_file}")


if __name__ == "__main__":
    main()
