# STN stimulation and EEG oscillations

## Subject processing: run these two pipelines in order

**For every participant, first run the posterior-channel pipeline; then run the all-channel pipeline.** Run on Mac for interactive QC.

From the repository root, for subject 101:

```bash
python analysis/subject/EEG_posterior_channels/run_subject_pipeline.py --subjects 101
python analysis/subject/EEG_all_channels/run_subject_pipeline.py --subjects 101 --platform mac
```

The posterior runner prompts for Mac/BlueBEAR and runs **BrainVision → BIDS → stimulation segmentation → cue epochs/manual rejection (PO3/PO4/POz) → ERP → TFR**. It establishes the raw BIDS data and participant report needed by the all-channel workflow.

The all-channel runner reads **existing raw BIDS**, then runs **posterior-alpha inclusion screening → continuous 1–100 Hz filtering/PyPREP/manual channel QC → stimulation segmentation/cue epochs → optional average reference → ICA → manual trial rejection → all-channel ERP/TFR → methods report**. It saves intermediate and cleaned FIF derivatives and JSON QC decisions. Bad channels remain marked, not interpolated.

For multiple subjects, replace `101` with e.g. `101 102 103` after `--subjects` in **both commands**. Do not start the all-channel pipeline for a subject until the posterior pipeline has finished successfully. If a participant is excluded at all-channel screening, subsequent all-channel stages are skipped.

Default all-channel reference is average; use `--rereference none` to keep the original reference. Mac is the default; `--platform bluebear` selects server paths, but interactive QC needs a working GUI.

## Group analyses

Run after subject processing is complete:

```bash
python analysis/group/EEG_posterior_channels/G02_grand_average_report.py --subjects 101 102 103
python analysis/group/EEG_all_channels/run_grand_average.py --subjects 101 102 103 --platform mac
```

The posterior group folder also provides `G01_concatenated_epochs_report.py` (trial-weighted rather than subject-weighted). The all-channel grand average uses only participants with each EEG sensor present and good in both conditions; per-sensor sample sizes may differ.

## Locations and details

- Raw organised recordings: `<PROJECT_ROOT>/data/data-organised/`
- BIDS raw and processed EEG: `<PROJECT_ROOT>/data/BIDS/` and `data/BIDS/derivatives/`
- Participant reports: `<PROJECT_ROOT>/derivatives/reports/sub-XXX/`
- Stimulation timing table: `analysis/subject/stimulation_cropped_time.json`

See the README in each subject/group analysis folder for commands, processing details and outputs. The workflows require interactive review; do not treat them as unattended batch jobs.

## Developer guide and PEP 8

See [code style, formatting, documentation and validation](docs/DEVELOPMENT.md). Run `python -m black --check analysis` and `python -m ruff check analysis` to audit the current code before reformatting. The legacy scripts are **not yet guaranteed PEP 8-compliant**; formatting and scientific validation must be reviewed together.
