# All-channel EEG preprocessing and subject analysis

This folder contains the full-montage EEG workflow. It is separate from the three-posterior-channel workflow in `analysis/subject/EEG_posterior_channels/`.

## Required input and BIDS behaviour

Stage 1 (`preprocessing/P01_epoch_pyprep_reref.py`) reads the subject's **existing raw EEG recording from BIDS** via MNE-BIDS. It does not read directly from `data/data-organised` and does not perform BrainVision-to-BIDS conversion.

The expected raw BIDS recording is organised under:

```text
data/BIDS/
  sub-XXX/
    ses-01/
      eeg/
        sub-XXX_ses-01_task-SpAtt_run-01_eeg.vhdr
        sub-XXX_ses-01_task-SpAtt_run-01_eeg.vmrk
        sub-XXX_ses-01_task-SpAtt_run-01_eeg.eeg
```

The exact extensions depend on the BrainVision recording, and the sidecar BIDS metadata files must also be present. If a participant has not yet been converted to BIDS, first run the posterior-channel runner documented in the repository-root README (that runner includes the original `P01_first_look_BIDS_conversion.py` stage), or convert and validate the raw recording with MNE-BIDS. The all-channel runner itself starts at QC/epoching, not raw BIDS conversion.

Stage 1 uses the crop times for `sub-XXX` in `analysis/subject/stimulation_cropped_time.json`. It filters the continuous data from 1 to 100 Hz before PyPREP, performs interactive channel review and PSD review, segments the conditions, creates cue-locked epochs (-0.5 to 1.6 s), and optionally average-references them.

## Processing order

1. **Screening (`P00_screen_subject.py`)**
   - Opens the existing participant PDF report for human inspection of posterior alpha modulation in PO3, PO4 and POz.
   - Saves the include/exclude decision to `data/BIDS/derivatives/sub-XXX/qc_all_channels/inclusion_decision.json`.
   - There is no automatic alpha threshold. The report needs to exist for useful screening.

2. **Filtering, PyPREP, manual channel review, segmentation, epoching and reference (`P01_epoch_pyprep_reref.py`)**
   - Reads the raw recording from BIDS with MNE-BIDS.
   - Filters continuous EEG from 1 to 100 Hz before PyPREP.
   - Shows suggested bad channels, allows manual edits, displays PSDs of rejected channels for possible rescue, and then displays the remaining-good-channel PSD for a final rejection decision.
   - Records the union of final bad channels and applies that same bad-channel list to the stimulation and no-stimulation conditions.
   - Creates cue-locked epochs from -0.5 to 1.6 s with `baseline=None`, `detrend=1`, and no automatic amplitude rejection.
   - Average-rereferences by default; pass `--rereference none` to retain the original reference.
   - Saves reference-aware epoch files and a JSON audit. This stage does not save new continuous stim/no-stim FIF files.

3. **ICA (`P02_ica.py`)**
   - Concatenates copies of the rereferenced stim and no-stim epochs.
   - Fits one ICA using an analysis copy resampled to 200 Hz and filtered 1–40 Hz.
   - Displays component maps and source time series for manual review.
   - Applies the same ICA solution and accepted component exclusions to both conditions.
   - Saves the ICA solution and component-decision JSON.

4. **Manual epoch and channel QC (`P03_manual_epoch_rejection.py`)**
   - Opens epochs for interactive bad-trial marking.
   - Allows additional channels to be marked bad with a recorded reason.
   - Saves final cleaned epochs and a JSON audit of epoch counts and final bad channels.

5. **Subject ERP/TFR (`sensor/A01_ERP.py`, `sensor/A02_TFR.py`, `sensor/A03_analysis_overview.py`)**
   - Calculates ERP and TFR analyses for the available good EEG channels.
   - Stim and no-stim TFR visualisations use percent baseline correction from -0.3 to -0.1 s.
   - The TFR difference and normalized ratio use the original un-baselined power: difference = stim - no-stim; ratio = (stim - no-stim) / (stim + no-stim).
   - The participant chooses the ERP/TFR ROI from available posterior/occipital candidate channels; this choice affects ROI summaries, not whether the other channels are computed.

## Run one subject

From the repository root on Mac:

```bash
python analysis/subject/EEG_all_channels/run_subject_pipeline.py --subjects 101 --platform mac
```

On BlueBEAR:

```bash
python analysis/subject/EEG_all_channels/run_subject_pipeline.py --subjects 101 --platform bluebear
```

The runner's subject input is plural: use `--subjects`, even for one subject. You can include several subjects:

```bash
python analysis/subject/EEG_all_channels/run_subject_pipeline.py --subjects 101 102 103 --platform mac
```

or an inclusive range:

```bash
python analysis/subject/EEG_all_channels/run_subject_pipeline.py --range 101 110 --platform mac
```

Average reference is the default. Choose no rereferencing with:

```bash
python analysis/subject/EEG_all_channels/run_subject_pipeline.py --subjects 101 --platform mac --rereference none
```

Resume at ICA (stage 2) only if screening and stage-1 outputs are already saved:

```bash
python analysis/subject/EEG_all_channels/run_subject_pipeline.py --subjects 101 --platform mac --from-stage 2
```

Stage numbers are 0 screening, 1 PyPREP/segmentation/epoching/reference, 2 ICA, 3 manual trial/channel QC, 4 ERP, 5 TFR and 6 analysis overview. Add `--continue-on-error` to continue to later subjects if one fails.

## Outputs and QC audit

- Final cleaned epochs: `data/BIDS/derivatives/sub-XXX/`, using names such as `..._no-stim_desc-avgref-clean_epo.fif` or `..._stim_desc-avgref-clean_epo.fif`. If `--rereference none` is used, the filenames identify `noref` instead.
- QC/audit JSON: `data/BIDS/derivatives/sub-XXX/qc_all_channels/`.
- ICA solution: `data/BIDS/derivatives/sub-XXX/qc_all_channels/sub-XXX_ica.fif`.
- Figures: `derivatives/figures/sub-XXX/EEG_all_channels/`.
- Participant PDF/report: `derivatives/reports/sub-XXX/`.

Bad channels are retained in the metadata and are not interpolated. Group analysis must skip each participant separately for each channel listed as bad or absent. That allows different electrodes to have different participant counts.

## Recommended validation before a large run

Run a participant whose data and artifacts you know well. Check the cue event counts and timing, PyPREP suggestions and PSDs, final bad-channel list, reference state, ICA component selection, epoch counts before/after manual rejection, and retained channel/bad-channel metadata. The pipeline is interactive; review and document each manual choice before processing the full cohort.
