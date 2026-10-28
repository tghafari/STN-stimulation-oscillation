# Python code style and documentation

## Safe formatting

Install the development tools in your existing analysis environment:

```bash
python -m pip install black ruff
```

Run these commands from the repository root, on a **new Git branch**:

```bash
git switch -c style/pep8-documentation
python -m black --check analysis
python -m ruff check analysis
```

To reformat the Python files (including archived scripts), then inspect changes:

```bash
python -m black analysis
git diff --check
git diff --stat
python -m compileall -q analysis
python -m ruff check analysis
```

Black changes formatting, not intended analysis logic. It does **not** guarantee behavioral equivalence; inspect the diff and test representative subjects before merging. Ruff checks are currently restricted to critical formatting, syntax and undefined-name classes because the repository contains legacy analysis scripts.

## Script documentation convention

Each active analysis script should start with a module docstring explaining:
- **Purpose:** scientific question and scope (subject/group, posterior/all channels).
- **Prerequisites:** upstream outputs, data layout, crop-time JSON and QC decisions.
- **Steps:** input loading, event/condition selection, filtering, reference, ICA, epoching, rejection, ERP/TFR/statistics and reporting, as applicable.
- **Parameters:** event alignment, time windows, frequency grid, baseline, contrast sign, ROI and group weighting.
- **Outputs:** exact file types and folders, plus interactive prompts and limitations.

Use one statement per line, four-space indentation, descriptive names, snake_case functions, imports at the top and a `main()` entry point when practical. Keep scientific transformations explicit rather than compressing multiple processing stages into one line.

## Scientific invariants to verify after style edits

- Posterior pipeline runs **first** and creates raw BIDS and participant report.
- All-channel pipeline reads raw BIDS, not posterior-cleaned epochs.
- Condition names remain `stim` and `no-stim`.
- Difference is `stim - no-stim`; normalized ratio is `(stim - no-stim)/(stim + no-stim)`.
- The ratio uses **unbaselined** power; descriptive stim/no-stim TFR plots use their documented percent baseline.
- ERP baseline, epoch timing, filtering, PyPREP, reference, ICA and manual decisions are unchanged.
- All-channel group analysis excludes bad/missing channels **per sensor** and reports different sample sizes where appropriate.

## Running pipelines

See [the root README](../README.md) for the two subject commands and the required order. See the corresponding folder README for stage-by-stage details and group commands.
