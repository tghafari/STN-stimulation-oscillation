"""Stage 2: fit ICA after rereferencing, inspect components manually, then apply ICA.

The implementation follows the logic of analysis/old/oldP06_run_apply_ICA.py but
removes subject-specific hard-coded component dictionaries. Instead, one ICA is
fit on concatenated stim + no-stim rereferenced continuous EEG, the component
plots are shown, and the researcher explicitly enters components to exclude.
That exact list is saved to JSON and the ICA solution itself is saved to FIF.

ICA is fit on a copy resampled to 200 Hz and filtered 1-40 Hz, as in the older
pipeline. The fitted ICA is then applied to the full-resolution rereferenced raw
and epochs for each condition.
"""
from __future__ import annotations

import argparse
import json

import mne
from mne.preprocessing import ICA

from pipeline_config import CONDITIONS, qc_dir, resolve_project_root, stage_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject", required=True)
    p.add_argument("--session", default="01")
    p.add_argument("--task", default="SpAtt")
    p.add_argument("--run", default="01")
    p.add_argument("--platform", choices=["mac", "bluebear"], default="mac")
    p.add_argument("--project-root", default=None)
    p.add_argument("--n-components", type=int, default=30)
    p.add_argument("--random-state", type=int, default=97)
    return p.parse_args()


def parse_component_list(text: str, n_components: int) -> list[int]:
    """Parse space/comma-separated component indices and validate the range."""
    cleaned = text.replace(",", " ").strip()
    if not cleaned:
        return []
    values = sorted(set(int(x) for x in cleaned.split()))
    bad = [x for x in values if x < 0 or x >= n_components]
    if bad:
        raise ValueError(f"Component indices outside 0..{n_components - 1}: {bad}")
    return values


def main() -> None:
    args = parse_args()
    subject = args.subject.removeprefix("sub-")
    root = resolve_project_root(args.platform, args.project_root)

    raws = {}
    epochs = {}
    for condition in CONDITIONS:
        raw_path = stage_path(root, subject, args.session, args.task, args.run, condition, "reref", "raw")
        epo_path = stage_path(root, subject, args.session, args.task, args.run, condition, "reref", "epo")
        if not raw_path.exists() or not epo_path.exists():
            raise FileNotFoundError(f"Missing Stage-1 output for {condition}: {raw_path} or {epo_path}")
        raws[condition] = mne.io.read_raw_fif(raw_path, preload=True)
        epochs[condition] = mne.read_epochs(epo_path, preload=True)

    # Concatenate copies so the original condition-specific objects stay untouched.
    fit_source = mne.concatenate_raws([raws[c].copy() for c in CONDITIONS], on_mismatch="warn")
    fit_source.pick("eeg")
    good_eeg = [ch for ch in fit_source.ch_names if ch not in fit_source.info["bads"]]
    if len(good_eeg) < 3:
        raise RuntimeError(f"Too few good EEG channels for ICA: {len(good_eeg)}")

    # Fast, ICA-specific copy; full-resolution data are not overwritten.
    fit_source.resample(200)
    fit_source.filter(1.0, 40.0)

    max_components = max(2, len(good_eeg) - 1)
    n_components = min(args.n_components, max_components)
    print(f"\nICA will use {n_components} components from {len(good_eeg)} good EEG channels.")
    print("Fit data: concatenated stim + no-stim, resampled 200 Hz, 1-40 Hz.")

    ica = ICA(
        method="fastica",
        random_state=args.random_state,
        n_components=n_components,
        max_iter="auto",
        verbose=True,
    )
    ica.fit(fit_source, reject_by_annotation=True, picks="eeg")

    # Human inspection: component maps plus time courses.
    ica.plot_components(show=True)
    ica.plot_sources(fit_source, block=True, title=f"sub-{subject}: inspect ICA components")

    while True:
        text = input(
            "\nICA components to EXCLUDE (e.g. 0 1 5, or Enter for none): "
        )
        try:
            excluded = parse_component_list(text, ica.n_components_)
            break
        except (ValueError, TypeError) as exc:
            print(f"Invalid component list: {exc}")

    print(f"Selected ICA components: {excluded or 'None'}")
    if excluded:
        # Show diagnostic properties before final confirmation.
        ica.plot_properties(fit_source, picks=excluded)
        answer = input("Apply this ICA exclusion list? [y/N]: ").strip().lower()
        if answer not in {"y", "yes"}:
            raise RuntimeError("ICA stage stopped by user before applying components.")

    ica.exclude = excluded
    ica_file = qc_dir(root, subject) / f"sub-{subject}_ica.fif"
    ica.save(ica_file, overwrite=True)

    audit = {
        "subject": f"sub-{subject}",
        "method": "fastica",
        "random_state": args.random_state,
        "fit_filter_hz": [1.0, 40.0],
        "fit_resample_hz": 200,
        "requested_n_components": args.n_components,
        "actual_n_components": int(ica.n_components_),
        "excluded_components": excluded,
        "ica_file": str(ica_file),
        "conditions": {},
    }

    for condition in CONDITIONS:
        raw_clean = raws[condition].copy()
        epo_clean = epochs[condition].copy()
        ica.apply(raw_clean)
        ica.apply(epo_clean)

        raw_out = stage_path(root, subject, args.session, args.task, args.run, condition, "ica", "raw")
        epo_out = stage_path(root, subject, args.session, args.task, args.run, condition, "ica", "epo")
        raw_clean.save(raw_out, overwrite=True)
        epo_clean.save(epo_out, overwrite=True)
        audit["conditions"][condition] = {
            "ica_raw": str(raw_out),
            "ica_epochs": str(epo_out),
            "bad_channels_retained_in_info": list(epo_clean.info["bads"]),
        }

    audit_file = qc_dir(root, subject) / "P02_ica_decisions.json"
    audit_file.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"Saved ICA decision audit: {audit_file}")


if __name__ == "__main__":
    main()
