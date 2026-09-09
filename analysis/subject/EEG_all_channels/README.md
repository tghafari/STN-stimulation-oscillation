# All-channel EEG preprocessing pipeline

This folder is a deliberately inspectable version of the single-subject EEG preprocessing pipeline. It reuses the logic and file conventions already present in this repository, especially:

- `analysis/subject/EEG_posterior_channels/` for current BIDS-derived segmentation and cue epoching conventions.
- `analysis/old/oldP06_run_apply_ICA.py` for the previous ICA strategy.

The new pipeline does **not** interpolate rejected channels. Bad channels remain in `info['bads']`, which is essential for the later group analysis where each electrode can be averaged over only the subjects in whom that electrode is good.

## Processing order

1. **Subject screening (`P00_screen_subject.py`)**
   - Opens the most recent existing participant HTML report.
   - You inspect alpha modulation in `PO3`, `PO4`, `POz`.
   - You explicitly choose include or exclude.
   - The decision is saved in `qc_all_channels/inclusion_decision.json`.
   - There is intentionally no automatic alpha-modulation threshold.

2. **Epoch definition + PyPREP + rereference (`P01_epoch_pyprep_reref.py`)**
   - Reads the existing BIDS-derived `stim` and `no-stim` continuous FIF files created by the repository's stimulation-segmentation step.
   - Defines cue-locked epochs from -0.5 to 1.6 s.
   - Runs PyPREP on the corresponding continuous data because PyPREP's noisy-channel detectors are intended for continuous EEG.
   - Shows the PSD so you can inspect the suggestions.
   - You can add channels or override PyPREP suggestions.
   - Takes the **union** of bad channels from stim and no-stim, then applies the same bad-channel set to both conditions.
   - Average-rereferences while excluding marked bad channels.
   - Saves rereferenced continuous data and rereferenced epochs.
   - Saves a complete JSON audit including detector reasons and your manual edits.

3. **ICA (`P02_ica.py`)**
   - Concatenates copies of the rereferenced stim and no-stim continuous data.
   - Fits ICA to EEG only after resampling the fitting copy to 200 Hz and filtering it 1-40 Hz, following the older repository ICA script.
   - Default request is 30 FastICA components, automatically capped if too few good EEG channels remain.
   - Opens ICA component maps and source time courses.
   - You manually enter components to exclude.
   - For non-empty exclusions, component properties are shown and you must confirm before application.
   - Saves the ICA solution and the exact excluded component list.
   - Applies the same ICA solution to both conditions.

4. **Manual trial rejection (`P03_manual_epoch_rejection.py`)**
   - Opens the ICA-cleaned epochs.
   - Displays all **good** EEG channels (scroll through them; default 20 visible at once).
   - You manually mark bad epochs in the MNE browser.
   - Saves final cleaned epochs and before/after epoch counts.

## Why the code keeps bad channels instead of dropping/interpolating them

Your planned group topomaps require channel-specific sample sizes. For example, if PO3 is good in 10 subjects but Cz is good in 12, the later group code should average PO3 across 10 and Cz across 12. To make that possible, this preprocessing pipeline leaves bad channels marked in `info['bads']` and records them in JSON. The later ERP/TFR group scripts must explicitly skip a participant for an electrode whenever that electrode appears in that participant's bad-channel list.

## Input expected by Stage 1

The pipeline currently expects the two continuous derivative files already produced by the repository's stimulation-segmentation workflow:

```text
.../data/BIDS/derivatives/sub-XXX/
    sub-XXX_ses-01_task-SpAtt_run-01_eeg_no-stim_raw.fif
    sub-XXX_ses-01_task-SpAtt_run-01_eeg_stim_raw.fif
```

This keeps the new code focused on the preprocessing steps requested here rather than duplicating the existing stimulation segmentation logic.

## Run one subject

From this folder:

```bash
python run_subject_pipeline.py --subjects 115 --platform mac
```

Or with an explicit project root:

```bash
python run_subject_pipeline.py \
    --subjects 115 \
    --platform mac \
    --project-root "/path/to/STN-in-PD"
```

## Run several subjects

```bash
python run_subject_pipeline.py --subjects 115 116 117 --platform mac
```

or an inclusive numeric range:

```bash
python run_subject_pipeline.py --range 115 123 --platform bluebear
```

Each subject is screened first. An excluded subject stops cleanly before preprocessing; the runner then moves to the next subject.

## Resume after a manual stopping point

```bash
python run_subject_pipeline.py --subjects 115 --platform mac --from-stage 2
```

Stage numbers are:

- `0` screening
- `1` epoch/PyPREP/rereference
- `2` ICA
- `3` manual epoch rejection

A saved inclusion decision of `include` is still required when resuming from Stage 1-3.

## Files you should inspect for every subject

Inside:

```text
data/BIDS/derivatives/sub-XXX/qc_all_channels/
```

check:

- `inclusion_decision.json`
- `P01_bad_channels_and_rereference.json`
- `sub-XXX_ica.fif`
- `P02_ica_decisions.json`
- `P03_manual_epoch_rejection.json`

The final cleaned epoch files are named with `desc-clean_epo.fif`.

## Deliberate safety / audit choices

- No source-code patching in memory.
- No automatic alpha inclusion decision.
- No automatic acceptance of PyPREP suggestions without showing the PSD and allowing corrections.
- No automatic ICA component rejection.
- No channel interpolation.
- Same bad-channel list and same ICA solution for stim and no-stim.
- Every manual decision is written to a small readable JSON file.
- Every major stage saves an intermediate FIF, so a result can be opened independently in MNE.

## Before using this for the whole dataset

Run it on **one participant you know very well** and compare each output against your previous processing. In particular verify:

1. event counts and cue timing;
2. PyPREP suggestions and PSDs;
3. `info['bads']` before/after rereferencing;
4. ICA maps, source time courses, and the effect of excluded components;
5. epoch counts before/after manual rejection;
6. that final stim/no-stim epochs contain the expected channel names and bad-channel metadata.

Only after those checks should you run a subject range.

## Next analysis stage (not included here yet)

The next scripts should compute subject-level ERP, TFR, and stimulation/no-stimulation TFR ratios from the final `desc-clean_epo.fif` files and then build channel-wise grand averages/topomaps that ignore a subject separately for each electrode listed as bad. That group code should also report the sample size `N` per electrode so the topomap is interpretable.
