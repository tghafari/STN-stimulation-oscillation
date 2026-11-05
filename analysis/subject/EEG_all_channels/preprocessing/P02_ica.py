"""Fit a shared ICA decomposition and manually review its components.

Prerequisite: P01 has produced reference-labelled, cue-locked epochs for
both DBS stimulation conditions. The original epochs are never resampled
or filtered for ICA fitting: only a concatenated fitting copy is.

Processing steps
----------------
1. Load stimulation ON and OFF epochs from P01.
2. Concatenate fitting copies, select EEG, resample to 200 Hz, and
   filter 1-40 Hz.
3. Fit one FastICA model (up to 30 components, configurable).
4. Display every component scalp map, then the ICA source browser.
5. Ask which components to exclude and optionally inspect their properties.
6. Apply the shared ICA solution to both original epoch sets.
7. Save condition-specific ICA-corrected epochs, the ICA solution, a
   participant report section, and an auditable JSON decision record.

No epoch rejection is performed here. P03 handles manual trial rejection.
"""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA

from all_channel_report import figure_dir, fmt_channels, participant_report
from pipeline_config import (
    CONDITIONS,
    qc_dir,
    resolve_project_root,
    stage_path,
)


def parse_args():
    """Read participant, BIDS run and reproducible ICA configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--session", default="01")
    parser.add_argument("--task", default="SpAtt")
    parser.add_argument("--run", default="01")
    parser.add_argument(
        "--platform", choices=["mac", "bluebear"], default="mac"
    )
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--n-components", type=int, default=30)
    parser.add_argument("--random-state", type=int, default=97)
    return parser.parse_args()


def parse_components(text, n_components):
    """Parse unique ICA indices and reject out-of-range selections."""
    values = (
        sorted(set(int(value) for value in text.replace(",", " ").split()))
        if text.strip()
        else []
    )
    if any(index < 0 or index >= n_components for index in values):
        raise ValueError(f"components must be 0..{n_components - 1}")
    return values


def main():
    """Fit ICA, collect manual exclusions and save corrected epochs."""
    args = parse_args()
    subject = args.subject.removeprefix("sub-")
    root = resolve_project_root(args.platform, args.project_root)
    report = participant_report(root, subject)
    figures = figure_dir(root, subject)
    epochs = {}

    # Load both conditions before fitting the shared ICA decomposition.
    for condition in CONDITIONS:
        path = stage_path(
            root, subject, args.session, args.task, args.run,
            condition, "reref", "epo",
        )
        if not path.exists():
            raise FileNotFoundError(
                f"Missing P01 epochs for {condition}: {path}"
            )
        epochs[condition] = mne.read_epochs(path, preload=True)

    # Only the ICA fitting copy is resampled and filtered. Applying the
    # fitted solution later preserves the original epoch time resolution.
    fit = mne.concatenate_epochs(
        [epochs[condition].copy() for condition in CONDITIONS],
        add_offset=True,
        on_mismatch="warn",
    )
    fit.pick("eeg")
    fit.resample(200)
    fit.filter(1, 40)

    good = [
        channel for channel in fit.ch_names
        if channel not in fit.info["bads"]
    ]
    n_components = min(args.n_components, max(2, len(good) - 1))
    print(
        f"Fitting one ICA with {n_components} components on "
        f"{len(fit)} concatenated stim + no-stim epochs."
    )
    ica = ICA(
        method="fastica",
        random_state=args.random_state,
        n_components=n_components,
        max_iter="auto",
    )
    ica.fit(fit, picks="eeg", reject_by_annotation=True)

    # The two manual review stages are intentionally sequential.
    print("\nICA MANUAL QC - STEP 1/2: COMPONENT SCALP MAPS")
    print(
        "Inspect all ICA component topographies. Close ALL "
        "component-map windows before continuing to the time series."
    )
    component_figures = ica.plot_components(
        picks=range(ica.n_components_), show=True
    )
    if not isinstance(component_figures, (list, tuple)):
        component_figures = [component_figures]
    for index, figure in enumerate(component_figures):
        report.add_figure(
            figure,
            str(figures / f"P02_all_ICA_components_{index + 1}.png"),
            f"All ICA components ({index + 1}/{len(component_figures)})",
            (
                f"All {ica.n_components_} fitted component topographies. "
                "No component is omitted."
            ),
            "ICA",
        )
    plt.show(block=True)

    print("\nICA MANUAL QC - STEP 2/2: COMPONENT TIME SERIES")
    print(
        "Inspect ICA source time series. Close the browser before "
        "selecting components for exclusion."
    )
    ica.plot_sources(
        fit,
        block=True,
        title=f"sub-{subject}: ICA component time series",
    )

    while True:
        try:
            excluded = parse_components(
                input(
                    "ICA components to EXCLUDE "
                    "(space/comma separated, Enter for none): "
                ),
                ica.n_components_,
            )
            break
        except ValueError as exc:
            print(exc)

    if excluded:
        properties = ica.plot_properties(
            fit, picks=excluded, show=False
        )
        if not isinstance(properties, (list, tuple)):
            properties = [properties]
        for index, figure in enumerate(properties):
            report.add_figure(
                figure,
                str(
                    figures / f"P02_ICA_exclusion_properties_{index + 1}.png"
                ),
                f"ICA exclusion diagnostic {index + 1}",
                f"Manually selected component(s): {excluded}.",
                "ICA",
            )
        answer = input(
            f"Apply exclusions {excluded}? [y/N]: "
        ).strip().lower()
        if answer not in {"y", "yes"}:
            raise RuntimeError("ICA stopped by user.")

    ica.exclude = excluded
    ica.save(
        qc_dir(root, subject) / f"sub-{subject}_ica.fif",
        overwrite=True,
    )
    audit = {
        "subject": f"sub-{subject}",
        "fit_on": "concatenated stim + no-stim epochs",
        "fit_resample_hz": 200,
        "fit_filter_hz": [1, 40],
        "n_components": int(ica.n_components_),
        "manual_qc_views": (
            "sequential: component scalp maps first, then source "
            "time series, then exclusion input"
        ),
        "excluded_components": excluded,
        "conditions": {},
    }

    for condition in CONDITIONS:
        cleaned = epochs[condition].copy()
        ica.apply(cleaned)
        output = stage_path(
            root, subject, args.session, args.task, args.run,
            condition, "ica", "epo",
        )
        cleaned.save(output, overwrite=True)
        audit["conditions"][condition] = {
            "n_epochs": len(cleaned),
            "bad_channels": list(cleaned.info["bads"]),
        }

    report_text = (
        "One shared ICA decomposition was fitted on concatenated "
        "stim + no-stim cue epochs.\n"
        "ICA fitting copy: resampled to 200 Hz and filtered 1-40 Hz.\n"
        f"Method: FastICA; random state: {args.random_state}.\n"
        f"Components fitted: {ica.n_components_}.\n"
        "Manual ICA QC was sequential: all component scalp maps were "
        "inspected first and closed; the component source time-series "
        "browser was then inspected and closed; component exclusions "
        "were entered only after both inspections.\n"
        f"Manually excluded components: {excluded or 'None'}.\n"
        "The fitted solution was then applied separately to the original "
        "stim epochs and no-stim epochs.\n"
        "Bad channels retained in info: "
        f"{fmt_channels(epochs[CONDITIONS[0]].info['bads'])}"
    )
    report.add_text("ICA method and decision", report_text, "ICA")
    (qc_dir(root, subject) / "P02_ica_decisions.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
