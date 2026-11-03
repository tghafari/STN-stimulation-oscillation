# Posterior EEG: STN stimulation laterality reports

This folder contains one runner, `run_stim_laterality_tfr.py`, that
generates **three PDF reports**. Each successfully analysed participant
contributes one page to each PDF.

## Input files

1. Clean, cue-locked MNE epoch files for `stim` and `no-stim`, from the
   existing EEG preprocessing pipeline.
2. `trial_alpha.csv`, produced by
   `analysis/subject/EEG_posterior_channels/block_preparation/B03_prepare_block_alpha.py`.
   It contains the verified mapping from the cleaned cue-epoch indices
   to the four confirmed stimulation periods (two OFF, one right STN ON,
   one left STN ON).
3. `individual_alpha_peaks.csv`, containing participant `subject` and
   `peak_hz`. By default, the runner first checks
   `derivatives/reports/group/individual_alpha_peaks.csv`, then the
   `EEG_posterior_channels_alpha_changes_stimOn_stimOff/` subdirectory.
   Use `--peaks-csv PATH` to choose a different file. Peaks are read
   as provided and are never re-estimated using ON data.

Run B03 again if the cleaned epoch files change, otherwise the
trial-to-period mapping may no longer be valid.

## Example command

From the repository root:

```bash
python analysis/group/EEG_posterior_channels/stim_laterality/run_stim_laterality_tfr.py \
    --subjects 101 102 103 105 106 107 108 112 113 114 115 116 118 120 121 123
```

Default platform is `mac`. Optional arguments:
`--platform bluebear`, `--project-root PATH`,
`--trial-csv PATH`, `--peaks-csv PATH`, `--output-dir PATH`,
`--session`, `--task`, `--run` and `--n-jobs`.

## The three PDFs

All are saved under
`<PROJECT_ROOT>/derivatives/reports/group/EEG_posterior_channels_stim_laterality/`.

### `posterior_TFR_right_STN_ON.pdf`

Every page analyses one participant's confirmed **right STN ON**
period compared with both OFF periods pooled. Five panels:

1. Right STN ON: left-versus-right sensor baseline-corrected alpha curves.
2. DBS OFF: left-versus-right sensor baseline-corrected alpha curves.
3. Left sensors: right STN ON minus OFF absolute-power TFR.
4. Right sensors: right STN ON minus OFF absolute-power TFR.
5. During right STN ON: right minus left sensor absolute-power TFR.

### `posterior_TFR_left_STN_ON.pdf`

Exactly the same layout, but the **left STN ON** period is compared with
both OFF periods. No right STN ON trials are included in the ON condition.

### `posterior_TFR_right_vs_left_STN_ON.pdf`

Every page directly compares the two ON periods **without OFF trials**.
Four panels:

1. Left sensors: right STN ON minus left STN ON absolute-power TFR.
2. Right sensors: right STN ON minus left STN ON absolute-power TFR.
3. Right sensors: right STN ON versus left STN ON baseline-corrected alpha time courses.
4. Left sensors: right STN ON versus left STN ON baseline-corrected alpha time courses.

## Sensors

The full posterior parent ROI consists of PO3, POz, PO4, O1, Oz, O2,
PO7, PO8. The lateralised comparisons use only:

- **Left sensors:** PO3, PO7, O1.
- **Right sensors:** PO4, PO8, O2.

POz and Oz are midline, therefore excluded from hemispheric comparisons.
A sensor must exist and be marked good in both ON periods and OFF to be
included. Every page prints the sensor names actually used.

## Frequency analysis and curves

Each epoch group is analysed using the same multitaper settings as the
existing G03 report: 2-31.5 Hz, 0.5 Hz spacing, cycles = frequency/2,
time-bandwidth = 2.0, decimation = 2. Power is averaged over retained
trials **before** averaging across the selected sensors.

The individual alpha curve is the mean linear power within the saved
participant peak **+/-1 Hz** at each cue-locked time sample. Curves
are displayed from -0.5 to 1.5 seconds relative to cue onset.
The shaded 0.3-0.8 second interval identifies the attention interval;
the curve itself is not restricted to this time window.

**Alpha curves are baseline corrected, but all TFR contrasts remain unbaselined.**
For each condition, multitaper power is averaged across trials FIRST, then
across the retained sensors. For each frequency, the condition-specific
mean power from -0.3 to -0.1 s is used to calculate percent change:
`100 * (power / baseline - 1)`. Frequency bins within the participant's
alpha peak +/-1 Hz are then averaged to form the curve. Alpha curves
are in percent baseline change, not microvolt squared. TFR differences
remain in linear microvolt squared (MNE V squared multiplied by 10^12).
TFR differences use the order given in their titles. Red means
a positive difference; blue means a negative difference.

The two ON-side TFRs and the within-ON sensor lateralisation TFR
share a symmetric zero-centred colour scale *within each participant
page*. The third report's two TFRs likewise share a scale. Colour
limits may vary across participants.

## Validation, limitations, and omissions

- Each PDF begins with a detailed methods page explaining trial selection,
frequency analysis, baseline correction and the meaning of every panel.
Each participant page reports their alpha peak, trial counts per condition
and the exact channel names used.

- The runner verifies that the G04/B03 trial indices cover all current
  cleaned cue epochs exactly once, and that all four periods and
  stimulation labels are consistent.
- Both left-cue and right-cue trials are pooled. The curves are not
  conditioned on the attended cue direction.
- The two OFF periods are pooled as the baseline-free comparator for
  each ON-side report. The direct ON-side comparison uses **no OFF** data.
- Incomplete or inconsistent participants are skipped with an explicit
  terminal message. No laterality assignment is guessed.
- These plots are descriptive; they do not test statistical significance
  or control for order/carryover effects.
