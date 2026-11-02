"""Validated BIDS cue-event to cleaned-Epochs matching.

Never infer trial block identity from Epochs sample indices without validating
against cue onset events in the original BIDS events.tsv. All cleaned epochs
must match uniquely; ambiguity or outdated crop times is a hard error.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd


def _cue_code(label):
    value = str(label).strip().lower().replace("-", "_").replace(" ", "_")
    if value == "cue_onset_left":
        return "left"
    if value == "cue_onset_right":
        return "right"
    return None


def build_expected(
    events, crop_ranges, task_blocks, sfreq, coordinate_system="concatenated"
):
    """Project BIDS cue events onto original or concatenated sample coordinates."""
    if coordinate_system not in ("original", "concatenated"):
        raise ValueError(f"Unknown coordinate system: {coordinate_system}")
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
            if coordinate_system == "original":
                sample = int(round(onset * sfreq))
            else:
                sample = int(round((offset + onset - crop_start) * sfreq))
            expected.append((sample, side, matched[0], onset))
        offset += crop_end - crop_start
    expected.sort(key=lambda x: x[0])
    if len({e[0] for e in expected}) != len(expected):
        raise ValueError("Duplicate projected BIDS cue samples; cannot map safely.")
    return expected


def _match_shift(observed, sides, expected, shift, tolerance):
    """Match every retained cue uniquely, in chronological order."""
    exp_samples = np.asarray([row[0] for row in expected], dtype=np.int64)
    exp_sides = [row[1] for row in expected]
    matches = []
    previous = -1
    for sample, side in zip(observed, sides):
        target = int(sample - shift)
        position = int(np.searchsorted(exp_samples, target))
        possible = [
            k for k in (position - 1, position, position + 1)
            if previous < k < len(expected)
            and exp_sides[k] == side
            and abs(int(exp_samples[k]) - target) <= tolerance
        ]
        if len(possible) != 1:
            return None
        previous = possible[0]
        matches.append(previous)
    return matches


def _diagnostic_rows(observed, sides, expected, shift):
    """Describe nearest same-cue timing residuals without claiming matches."""
    rows = []
    for index, (sample, side) in enumerate(zip(observed, sides)):
        choices = [
            (abs(int(sample) - int(item[0]) - shift), j)
            for j, item in enumerate(expected) if item[1] == side
        ]
        nearest = min(choices) if choices else None
        matched = expected[nearest[1]] if nearest else None
        rows.append({
            "clean_epoch_index": index,
            "clean_event_sample": int(sample),
            "cue_direction": side,
            "nearest_bids_onset_s": matched[3] if matched else np.nan,
            "nearest_bids_block_index": matched[2] if matched else np.nan,
            "nearest_residual_samples": nearest[0] if nearest else np.nan,
            "candidate_shift_samples": shift,
            "verified": False,
        })
    return rows


def validate_mapping(
    epochs, expected, condition, tolerance_samples=2,
    diagnostic_csv=None,
):
    """Validate a unique BIDS cue assignment; save diagnostics on failure.

    Search all possible same-direction event pairs, not just the first
    20 cleaned and 50 BIDS events. A valid offset must match EVERY
    retained cue uniquely and in chronological order within two samples.

    Per-crop discontinuities are investigated in the diagnostic CSV but
    never accepted automatically: a piecewise mapping must be separately
    established and validated before block-level inference.
    """
    if not expected:
        raise ValueError(f"{condition}: no BIDS cues in crop intervals")
    observed = epochs.events[:, 0].astype(np.int64)
    inverse = {int(code): _cue_code(name)
               for name, code in epochs.event_id.items()}
    sides = [inverse.get(int(code)) for code in epochs.events[:, 2]]
    if any(side is None for side in sides):
        raise ValueError(f"{condition}: non-cue event in cleaned epochs")
    if len(observed) > len(expected):
        raise ValueError(f"{condition}: more cleaned than BIDS cues")

    candidates = {
        int(sample - item[0])
        for sample, side in zip(observed, sides)
        for item in expected if item[1] == side
    }
    solutions = {}
    for shift in sorted(candidates):
        matches = _match_shift(
            observed, sides, expected, shift, tolerance_samples
        )
        if matches is not None:
            solutions.setdefault(tuple(matches), []).append(shift)
            if len(solutions) > 1:
                break

    if len(solutions) != 1:
        # Select the candidate matching the largest number of early
        # events for diagnostics ONLY, never for block assignment.
        scored = []
        for shift in candidates:
            count = 0
            for sample, side in zip(observed, sides):
                if any(
                    item[1] == side
                    and abs(int(sample - item[0] - shift))
                    <= tolerance_samples
                    for item in expected
                ):
                    count += 1
            scored.append((count, shift))
        best_count, best_shift = max(scored) if scored else (0, 0)
        if diagnostic_csv is not None:
            path = Path(diagnostic_csv)
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                _diagnostic_rows(observed, sides, expected, best_shift)
            ).to_csv(path, index=False)
        raise ValueError(
            f"{condition}: {len(solutions)} valid global offsets; "
            f"best candidate explains {best_count}/{len(observed)} "
            f"cue timings within {tolerance_samples} samples. "
            f"Diagnostic CSV: {diagnostic_csv}. "
            "Do not assign blocks until timing is uniquely verified."
        )

    matches_key, shifts = next(iter(solutions.items()))
    matches = list(matches_key)
    shift = min(
        shifts,
        key=lambda value: (
            max(
                abs(int(sample) - int(expected[index][0]) - value)
                for sample, index in zip(observed, matches)
            ),
            sum(
                abs(int(sample) - int(expected[index][0]) - value)
                for sample, index in zip(observed, matches)
            ),
            abs(value),
        ),
    )
    return {
        "sample_offset": int(shift),
        "equivalent_sample_offsets": sorted(shifts),
        "matched_expected_indices": matches,
        "matched_bids_onsets_s": [float(expected[i][3]) for i in matches],
        "matched_task_block_indices": [int(expected[i][2]) for i in matches],
        "n_bids_cues_in_crops": len(expected),
        "n_clean_epochs": len(observed),
        "n_unretained_bids_cues": len(expected) - len(observed),
    }

def map_condition_epochs(
    epochs, events, crop_ranges, task_blocks, condition, diagnostic_csv=None
):
    """Try both event coordinate systems without accepting ambiguous mappings.

    An original-coordinate mapping uses BIDS onset * sampling frequency.
    A concatenated-coordinate mapping subtracts crop starts and accumulates
    retained crop durations. Both undergo identical strict one-to-one
    validation. If both succeed they must identify identical BIDS events.
    """
    solutions = {}
    failures = {}
    for system in ("original", "concatenated"):
        expected = build_expected(
            events, crop_ranges, task_blocks,
            float(epochs.info["sfreq"]), coordinate_system=system,
        )
        output = None
        if diagnostic_csv is not None:
            path = Path(diagnostic_csv)
            output = path.with_name(f"{path.stem}_{system}{path.suffix}")
        try:
            solutions[system] = validate_mapping(
                epochs, expected, condition, diagnostic_csv=output
            )
        except ValueError as exc:
            failures[system] = str(exc)

    if not solutions:
        raise ValueError(
            f"{condition}: both timing coordinate systems failed. "
            f"Original: {failures.get('original')}. "
            f"Concatenated: {failures.get('concatenated')}."
        )
    if len(solutions) == 2:
        original = solutions["original"]["matched_bids_onsets_s"]
        concatenated = solutions["concatenated"]["matched_bids_onsets_s"]
        if original != concatenated:
            raise ValueError(
                f"{condition}: original and concatenated systems produce "
                "different BIDS cue assignments; mapping is ambiguous."
            )

    selected = (
        "original" if "original" in solutions else "concatenated"
    )
    result = solutions[selected]
    result["coordinate_system"] = selected
    print(f"{condition}: validated using {selected} coordinates")
    return result
