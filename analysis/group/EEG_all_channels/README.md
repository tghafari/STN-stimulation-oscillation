# All-channel EEG group analysis

## Overview

This folder contains the group-level ERP, TFR, alpha-power quality-control, and cluster-permutation analyses for the STN stimulation study.

All scripts use the final cleaned EEG epochs from each participant, separately for stimulation ON and OFF. Attention-left and attention-right cue epochs are combined.

## Scripts

| Script | Description |
|---|---|
| `G01_complete_grand_average_report.py` | Main group-analysis implementation. Calculates grand-average ERP and TFR, sensor-level stimulation contrasts, and posterior ROI summaries. The plotted stimulation-minus-no-stimulation TFR is the difference of condition-wise baseline-percent changes; the normalized ratio remains unbaselined. |
| `run_grand_average.py` | Recommended entry point for G01. Adds detailed scalp and posterior figures to the group PDF. |
| `Q01_participant_eight_posterior_TFR_QC.py` | Located in `analysis/quality_controls/`. Participant posterior TFR grids, 10–12 Hz trial-level alpha QC, and OFF-derived individual alpha peak plots/CSV (former posterior G02). |
| `Q02_alpha_sensor_scalp_QC.py` | Descriptive alpha-power time courses across EEG sensors. Displays group mean ± SEM, using the participants retaining each sensor. |
| `S01_alpha_cluster_report.py` | Paired cluster-permutation statistics for alpha-power stimulation effects, including individual sensors and posterior ROIs. |

## Running the analyses

Open a terminal in the repository root.


### 1. Grand-average ERP and TFR

```bash
python analysis/group/EEG_all_channels/run_grand_average.py --subjects 101 102 103 104 105 106 107 108 110 111 112 113 114 115 116 118 119 120 121 122 123 --platform mac
```

This runs G01 internally. Do not run both scripts separately unless specifically debugging G01.

### 2. Q01 — Individual-participant posterior TFR QC

```bash
python analysis/group/EEG_all_channels/Q01_participant_eight_posterior_TFR_QC.py --subjects 101 102 103 104 105 106 107 108 110 111 112 113 114 115 116 118 119 120 121 122 123 --platform mac
```

Also exports `derivatives/reports/group/EEG_posterior_channels_alpha_changes_stimOn_stimOff/individual_alpha_peaks.csv`, required for laterality. Creates participant grids for three posterior channels (PO3, POz, PO4) and eight posterior channels (PO3, POz, PO4, O1, Oz, O2, PO7, PO8). No averaging across participants.

### 3. Q02 — Alpha-power scalp QC

```bash
python analysis/group/EEG_all_channels/Q02_alpha_sensor_scalp_QC.py --subjects 101 102 103 104 105 106 107 108 110 111 112 113 114 115 116 118 119 120 121 122 123 --platform mac --alpha 10 12
```

Produces alpha-power time courses for individual scalp sensors, with group mean ± SEM. Participants with a missing or bad channel are excluded only from that channel's summary.

**Important:** The current Q02 implementation must be checked against its baseline documentation before scientific interpretation.

### 4. S01 — Alpha cluster-permutation statistics

```bash
python analysis/group/EEG_all_channels/S01_alpha_cluster_report.py --subjects 101 102 103 104 105 106 107 108 110 111 112 113 114 115 116 118 119 120 121 122 123 --platform mac --alpha 10 12
```

Tests participant-level stimulation-minus-no-stimulation alpha-power differences over 0.2–1.2 seconds after cue onset.

The inferential contrast uses unbaselined power. Cluster tests are performed for individual sensors and predefined posterior ROIs.

Optional `--sensor-fdr` applies FDR correction across single-sensor cluster results.

## Analysis conventions

- ERP baseline: −0.1 to 0 seconds.
- Descriptive TFR baseline: −0.3 to −0.1 seconds, percent change within each condition.
- Displayed TFR difference: stimulation percent change minus no-stimulation percent change, each baseline-corrected over −0.3 to −0.1 seconds per participant and sensor (percentage-point difference). This avoids dividing a near-zero raw difference by its own baseline.
- Normalized TFR difference `(stim - no-stim) / (stim + no-stim)`: original unbaselined power, **no baseline correction**.
- Absolute alpha-power QC difference: original unbaselined power (not the plotted baseline-corrected TFR difference).
- Three-channel ROI: PO3, POz, PO4.
- Eight-channel ROI: PO3, POz, PO4, O1, Oz, O2, PO7, PO8.
- Bad or missing channels are not interpolated for group analysis.
- TFR scalp maps: average participants independently for each sensor (excluding participants missing that sensor).
- TFR ROI plots: average available good ROI sensors **within each participant first**, then average participant-level ROI arrays equally. Participants with no usable ROI sensors are excluded from that ROI. ROI channel membership can differ across participants and is documented in the report.
- ERP ROI plots and separate Q01/Q02/S01 analyses are unchanged by this G01 TFR-only update.
- The subject is the unit of statistical inference.

## Outputs

- Main group PDF: `derivatives/reports/group/EEG_all_channels_complete_grand_average/`
- Q01 PDF: `derivatives/reports/QC/posterior_TFR_all_subjects/`
- Q02 figures: `derivatives/reports/group/EEG_all_channels_alpha_QC/`
- S01 statistical PDF: `derivatives/reports/group/EEG_all_channels_alpha_cluster/`

## Recommended order

Run `Q01` first to inspect individual-participant TFRs, followed by `Q02` for descriptive alpha-power QC. Run `run_grand_average.py` for the main group figures and `S01` for statistical inference.

These scripts should not be deleted: they perform different, complementary analyses.
## Individual-alpha-peak-aligned TFR report

Running `run_grand_average.py` (or G01 directly) now produces a **second PDF** alongside the standard grand-average report. It reuses G01's subject-level TFRs; it does not recompute them.

For each participant, the alpha peak is estimated from unbaselined no-stimulation power, averaged across available good ROI8 posterior channels and 0.3-0.8 s after cue onset. The largest value in 8-12 Hz is selected at the existing 0.5-Hz frequency resolution, matching `analysis/quality_controls/Q01_participant_eight_posterior_TFR_QC.py`.

Each participant's frequency axis is shifted to `frequency - individual alpha peak`, interpolated to a common -6 to +6 Hz relative-frequency grid, and then averaged across participants. The y-axis is relative frequency: 0 Hz is the individual alpha peak, negative values are below it, and positive values are above it. Group sensor averages use only participants with that sensor good in both conditions. ROI figures average available ROI sensors within each participant before averaging participants equally.

- Stim and no-stim: percent baseline corrected per participant/sensor (-0.3 to -0.1 s).
- Difference: stim percent-baseline change minus no-stim percent-baseline change (percentage points).
- Ratio: (stim - no-stim) / (stim + no-stim), calculated from original unbaselined power.

The additional report is named `complete_grand_average_<subjects>_peak_aligned.pdf`. Its figures are saved in the report's `figures/peak_aligned/` folder and its peak estimates, interpolation grid, and channel membership are recorded in a companion JSON file under the all-channel group derivatives folder.

This peak is an OFF-derived maximum-power estimate, not an aperiodic-corrected oscillatory peak. It is restricted to the 8-12 Hz range and should be interpreted as the operational individual alpha frequency used for alignment.
