# STN stimulation and EEG oscillation analysis

Analysis code for the effect of STN stimulation on EEG oscillatory activity.

## Choose the subject pipeline

There are two distinct single-subject EEG workflows. They are not interchangeable.

### 1. Posterior-channel workflow (PO3, PO4, POz)

This is the older workflow. It converts the organised BrainVision recording to BIDS, segments stimulation and no-stimulation periods, epochs the data, and runs the three-channel ERP/TFR analyses.

From the repository root, run one subject:

```bash
python analysis/subject/EEG_posterior_channels/run_subject_pipeline.py --subjects 101
```

The script asks whether you are running on BlueBEAR or Mac. It then runs the full posterior-channel workflow, including interactive steps. To process several subjects:

```bash
python analysis/subject/EEG_posterior_channels/run_subject_pipeline.py --subjects 101 102 103
```

Or an inclusive numeric range:

```bash
python analysis/subject/EEG_posterior_channels/run_subject_pipeline.py --range 101 110
```

**BIDS output:** the first stage, `preprocessing/P01_first_look_BIDS_conversion.py`, reads the BrainVision data from `data/data-organised`, adds/corrects event annotations, and calls MNE-BIDS to write the raw recording into `data/BIDS`. The runner then continues with segmentation, epoching, ERP and TFR. This workflow runs the entire sequence; the runner does not provide a command-line option to run P01 alone.

### 2. All-channel workflow

Use this workflow for analysis across the full EEG montage, with PyPREP/manual bad-channel review, rereferencing, ICA, manual bad-epoch rejection, and all-channel ERP/TFR analysis.

**Prerequisite:** a readable raw EEG recording must already exist in the BIDS dataset at `data/BIDS`. The all-channel pipeline reads the BIDS recording; it does **not** perform the original BrainVision-to-BIDS conversion itself. If subject 101 has not been converted to BIDS yet, run the posterior-channel workflow above first, or otherwise create/verify the subject's BIDS raw recording before starting this workflow. The all-channel workflow also starts with an inclusion-screening step that opens the existing participant report.

From the repository root on Mac:

```bash
python analysis/subject/EEG_all_channels/run_subject_pipeline.py --subjects 101 --platform mac
```

On BlueBEAR:

```bash
python analysis/subject/EEG_all_channels/run_subject_pipeline.py --subjects 101 --platform bluebear
```

Several subjects:

```bash
python analysis/subject/EEG_all_channels/run_subject_pipeline.py --subjects 101 102 103 --platform mac
```

Inclusive numeric range:

```bash
python analysis/subject/EEG_all_channels/run_subject_pipeline.py --range 101 110 --platform mac
```

Average rereferencing is the default. To keep the reference from acquisition, pass `--rereference none`:

```bash
python analysis/subject/EEG_all_channels/run_subject_pipeline.py --subjects 101 --platform mac --rereference none
```

The stages are:
0. Inspect the participant report and record the inclusion decision.
1. Filter the continuous BIDS EEG, run PyPREP and interactive channel/PSD review, segment using `analysis/subject/stimulation_cropped_time.json`, create cue-locked epochs, and optionally average-reference.
2. Fit one ICA solution across stimulation and no-stimulation epochs, inspect components, and apply the same exclusions to both conditions.
3. Manually reject bad trials, optionally mark additional bad channels, and save final cleaned epochs.
4. Calculate participant ERP.
5. Calculate participant TFR and the raw-power stimulation/no-stimulation normalized ratio.
6. Add the participant analysis overview to the report.

Resume at a stage only after the required previous-stage outputs and the saved `include` decision exist. For example, resume at ICA:

```bash
python analysis/subject/EEG_all_channels/run_subject_pipeline.py --subjects 101 --platform mac --from-stage 2
```

Stage numbers are 0–6 as listed above. To continue to later subjects after a subject fails, add `--continue-on-error`.

The all-channel outputs are written under the project's `data/BIDS/derivatives/sub-101/` folder and participant reports/figures under `derivatives/reports/sub-101/` and `derivatives/figures/sub-101/EEG_all_channels/`. Rejected channels are recorded as bad and are not interpolated, so the later group analysis can calculate each channel using the subjects in whom that channel is good.

## Group analysis

The all-channel group report entry point is:

```bash
python analysis/group/EEG_all_channels/run_grand_average.py --subjects 101 102 103 --platform mac
```

Use `--platform bluebear` on BlueBEAR. Check `analysis/group/EEG_all_channels/README.md` for current group outputs and interpretation.

## Notes

- Run the scripts from the repository root unless a README specifies otherwise.
- The workflows contain interactive QC; they are not fully unattended batch jobs.
- Keep the JSON QC/audit files and the chosen subject list with the outputs for reproducibility.
- Verify the subject's BIDS raw files and stimulation crop-time entry before running either workflow.
