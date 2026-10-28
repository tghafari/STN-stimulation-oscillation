"""Validated BIDS cue-event to cleaned-Epochs matching.

Never infer trial block identity from Epochs sample indices without validating
against cue onset events in the original BIDS events.tsv. All cleaned epochs
must match uniquely; ambiguity or outdated crop times is a hard error.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _cue_code(label):
    value = str(label).strip().lower().replace("-", "_").replace(" ", "_")
    if value == "cue_onset_left":
        return "left"
    if value == "cue_onset_right":
        return "right"
    return None


def build_expected(events, crop_ranges, task_blocks, sfreq):
    """Map BIDS cue events into concatenated-condition sample coordinates."""
    if "trial_type" not in events or "onset" not in events:
        raise ValueError("BIDS events.tsv lacks trial_type or onset.")
    events = events.copy()
    events["onset"] = pd.to_numeric(events["onset"], errors="coerce")
    events = events.dropna(subset=["onset"])
    expected = []
    offset = 0.0
    for crop_start, crop_end in crop_ranges:
        for _, event in events.iterrows():
            onset = float(event["onset"])
            side = _cue_code(event["trial_type"])
            if side is None or not (crop_start <= onset <= crop_end):
                continue
            matched = [i for i, (start, end) in enumerate(task_blocks)
                       if start <= onset <= end]
            if len(matched) != 1:
                raise ValueError(
                    f"Cue at {onset:.3f}s is in a crop but not uniquely inside a BIDS task block."
                )
            expected.append((int(round((offset + onset - crop_start) * sfreq)),
                             side, matched[0], onset))
        offset += crop_end - crop_start
    expected.sort(key=lambda x: x[0])
    if len({e[0] for e in expected}) != len(expected):
        raise ValueError("Duplicate projected BIDS cue samples; cannot map safely.")
    return expected


def validate_mapping(epochs, expected, condition, tolerance_samples=2):
    """Find and validate a single sample-origin offset, never assume its value."""
    if not expected:
        raise ValueError(f"{condition}: no BIDS cue events in the saved crop intervals.")
    observed_samples = epochs.events[:, 0].astype(np.int64)
    inverse_ids = {int(v): _cue_code(k) for k, v in epochs.event_id.items()}
    observed_sides = [inverse_ids.get(int(x)) for x in epochs.events[:, 2]]
    if any(x is None for x in observed_sides):
        raise ValueError(f"{condition}: cleaned epochs contain non-cue event IDs.")
    exp_samples = np.asarray([x[0] for x in expected], dtype=np.int64)
    exp_sides = [x[1] for x in expected]
    if len(observed_samples) > len(expected):
        raise ValueError(f"{condition}: {len(observed_samples)} cleaned epochs exceed "
                         f"{len(expected)} BIDS cue events inside crops. Check the JSON.")
    # The sample origin of concatenated Raw depends on first_samp and crop
    # operations. Test candidate offsets and require unique cue identity and
    # near-exact timing for every retained epoch.
    candidates = set()
    for i in range(min(20, len(observed_samples))):
        for j in range(min(50, len(expected))):
            if observed_sides[i] == exp_sides[j]:
                candidates.add(int(observed_samples[i] - exp_samples[j]))
    results = []
    for shift in candidates:
        matches = []
        used = set()
        for sample, side in zip(observed_samples, observed_sides):
            target = int(sample - shift)
            ix = int(np.searchsorted(exp_samples, target))
            neighbors = [k for k in (ix-1, ix, ix+1) if 0 <= k < len(expected)]
            possible = [k for k in neighbors if k not in used
                        and exp_sides[k] == side
                        and abs(int(exp_samples[k]) - target) <= tolerance_samples]
            if len(possible) != 1:
                break
            k = possible[0]
            used.add(k)
            matches.append(k)
        if len(matches) == len(observed_samples):
            results.append((shift, matches))
    if len(results) != 1:
        raise ValueError(
            f"{condition}: could not uniquely verify BIDS-to-clean-epoch timing "
            f"({len(observed_samples)} retained, {len(expected)} BIDS cues, "
            f"{len(results)} valid sample offsets). Crop JSON may differ from "
            "the version used to generate the cleaned epochs. Re-run preprocessing "
            "with corrected crop times; do not use unverified TFR results."
        )
    shift, matches = results[0]
    if matches != sorted(matches):
        raise ValueError(f"{condition}: mapped events are not chronologically ordered.")
    return {
        "sample_offset": int(shift),
        "matched_expected_indices": matches,
        "matched_bids_onsets_s": [float(expected[i][3]) for i in matches],
        "matched_task_block_indices": [int(expected[i][2]) for i in matches],
        "n_bids_cues_in_crops": len(expected),
        "n_clean_epochs": len(observed_samples),
        "n_unretained_bids_cues": len(expected) - len(observed_samples),
    }


def map_condition_epochs(epochs, events, crop_ranges, task_blocks, condition):
    expected = build_expected(events, crop_ranges, task_blocks, float(epochs.info["sfreq"]))
    return validate_mapping(epochs, expected, condition)
