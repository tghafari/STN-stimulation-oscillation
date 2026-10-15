# All-channel subject pipeline (run second)

**Prerequisite:** finish the posterior-channel subject pipeline for the same participant first. That pipeline performs BrainVision-to-BIDS conversion and creates the participant report. This pipeline reopens the raw BIDS recording; it does **not** reuse posterior-only cleaned epochs or perform BIDS conversion.

From the repository root:

```bash
python analysis/subject/EEG_all_channels/run_subject_pipeline.py --subjects 101 --platform mac
```

Use `--subjects 101 102 103` for several subjects or `--range 101 110` for an inclusive range. `--platform bluebear` uses BlueBEAR paths, but interactive QC needs a functioning GUI. Average reference is the default; add `--rereference none` to preserve the acquisition reference.

## Stages (0–6)

0. **P00 screening:** open the existing participant report, visually assess PO3/PO4/POz alpha modulation, and save the human include/exclude decision.
1. **P01 PyPREP/epochs/reference:** read raw BIDS, filter continuous data 1–100 Hz, detect/review noisy channels with PyPREP and PSDs, use `stimulation_cropped_time.json` to segment stim/no-stim, epoch cue onset (-0.5 to 1.6 s; no baseline; detrend=1), mark the same bad channels in both conditions and optionally average-reference. Save epochs and QC JSON.
2. **P02 ICA:** concatenate condition epochs for a shared FastICA fit (200-Hz, 1–40-Hz fitting copy; up to 30 components by default), inspect maps/sources, choose excluded components, apply the same ICA to both conditions, and save ICA/audit.
3. **P03 manual epoch QC:** reject bad trials interactively, optionally mark further bad channels, inspect PSD and save final cleaned epochs and counts.
4. **A01 ERP:** combine attention-left/right within condition; average good EEG channels, 30-Hz low-pass, baseline -0.1 to 0 s; inspect/select posterior ROI and save evokeds/figures.
5. **A02 TFR:** multitaper 2–30 Hz in 1-Hz steps, cycles = frequency/2, time-bandwidth=2, decim=2; percent baseline (-0.3 to -0.1 s) for stim/no-stim plots; raw-power stim-minus-no-stim difference and normalized ratio; save TFRs and ROI decisions.
6. **A03 overview:** append an auditable methods summary to the participant PDF.

Resume at ICA only after stages 0–1 succeeded:

```bash
python analysis/subject/EEG_all_channels/run_subject_pipeline.py --subjects 101 --platform mac --from-stage 2
```

Use `--continue-on-error` to move to later subjects after a failure.

## Outputs

- Raw BIDS input: `<PROJECT_ROOT>/data/BIDS/sub-XXX/ses-01/eeg/`
- Cleaned epochs/ERP/TFR: `<PROJECT_ROOT>/data/BIDS/derivatives/sub-XXX/`
- QC JSON and ICA: `.../derivatives/sub-XXX/qc_all_channels/`
- PDF: `<PROJECT_ROOT>/derivatives/reports/sub-XXX/`

Bad channels remain marked (not interpolated). The group pipeline averages each sensor over only participants in whom it is good in both conditions. Group command:

```bash
python analysis/group/EEG_all_channels/run_grand_average.py --subjects 101 102 103 --platform mac
```
