# All-channel EEG group analysis

## Overview

This folder contains the group-level ERP, TFR, alpha-power quality-control, and cluster-permutation analyses for the STN stimulation study.

All scripts use the final cleaned EEG epochs from each participant, separately for stimulation ON and OFF. Attention-left and attention-right cue epochs are combined.

## Scripts

| Script | Description |
|---|---|
| `G01_complete_grand_average_report.py` | Main group-analysis implementation. Calculates grand-average ERP and TFR, sensor-level stimulation contrasts, and posterior ROI summaries. The plotted stimulation-minus-no-stimulation TFR is the difference of condition-wise baseline-percent changes; the normalized ratio remains unbaselined. |
| `run_grand_average.py` | Recommended entry point for G01. Adds detailed scalp and posterior figures to the group PDF. |

## Running the analyses

Open a terminal in the repository root.


### 1. Grand-average ERP and TFR

```bash
python analysis/group/EEG_all_channels/run_grand_average.py --subjects 101 102 103 104 105 106 107 108 110 111 112 113 114 115 116 118 119 120 121 122 123 --platform mac
```

This runs G01 internally. Do not run both scripts separately unless specifically debugging G01.

## Analysis conventions

- ERP baseline: −0.1 to 0 seconds.
- Descriptive TFR baseline: −0.3 to −0.1 seconds, percent change within each condition.
- Displayed TFR difference: stimulation percent change minus no-stimulation percent change, each baseline-corrected over −0.3 to −0.1 seconds per participant and sensor (percentage-point difference). This avoids dividing a near-zero raw difference by its own baseline.
- Normalized TFR difference `(stim - no-stim) / (stim + no-stim)`: original unbaselined power, **no baseline correction**.
- Absolute alpha-power QC difference: original unbaselined power (not the plotted baseline-corrected TFR difference).
- Three-channel ROI (ROI3): PO3, POz, PO4.
- Six-channel ROI (ROI6): PO3, PO4, O2, O1, POz, Oz.
- Eight-channel ROI (ROI8): PO3, POz, PO4, O1, Oz, O2, PO7, PO8.
- Bad or missing channels are not interpolated for group analysis.
- TFR scalp maps: average participants independently for each sensor (excluding participants missing that sensor).
- TFR ROI plots: average available good ROI sensors **within each participant first**, then average participant-level ROI arrays equally. Participants with no usable ROI sensors are excluded from that ROI. ROI channel membership can differ across participants and is documented in the report.
- ERP ROI plots and separate Q01/Q02/S01 analyses are unchanged by this G01 TFR-only update.
- The subject is the unit of statistical inference.

## Outputs

- Main group PDF: `derivatives/reports/group/EEG_all_channels_complete_grand_average/`

## Recommended order

Run `Q01` first to inspect individual-participant TFRs, followed by `Q02` for descriptive alpha-power QC. Run `run_grand_average.py` for the main group figures and `S01` for statistical inference.

These scripts should not be deleted: they perform different, complementary analyses.
## Individual-alpha-peak-aligned TFR report (second PDF)

**Required input:** First run `analysis/quality_controls/Q01_participant_eight_posterior_TFR_QC.py` with the same participants to create `derivatives/reports/group/EEG_posterior_channels_alpha_changes_stimOn_stimOff/individual_alpha_peaks.csv`. G01 reads the `peak_hz` column directly. It does **not** re-estimate peaks or round them to integers: 8.5 Hz remains 8.5 Hz. If a requested participant is missing, the script stops before expensive TFR computation.

The standard G01 command also generates a second, peak-aligned PDF. Its relative-frequency y-axis is set by `RELATIVE_FREQS` (currently **-6 to +18 Hz**): 0 Hz corresponds to each participant's CSV alpha peak. Subject-level TFRs are interpolated onto this relative-frequency grid **before** group averaging. Per-sensor grand averages use the participants with that sensor present and good in both conditions. ROI3/ROI8 grand averages average available channels within participant first.

In **both the normal-frequency and peak-aligned TFR reports**:

- Stim ON and stim OFF: each participant/sensor is percent-baseline corrected over -0.3 to -0.1 s before group averaging.
- Difference: baseline-correct ON and OFF **separately**, calculate ON minus OFF in percentage points for each participant/sensor, then grand-average.
- Ratio: calculate `(ON - OFF)/(ON + OFF)` from **unbaselined** power, then grand-average.

The additional **absolute-power QC** table reports unbaselined 8-12 Hz power and its raw difference only to check power values and condition labels. It is **not** the plotted baseline-corrected TFR difference.

The peak-aligned report and its `figures/peak_aligned/` folder are saved beside the standard report under `derivatives/reports/group/EEG_all_channels_complete_grand_average/`. A companion analysis JSON records the CSV source, exact peaks, relative-frequency grid and per-channel participant membership.

## ROI6 outputs

G01 now includes ROI6 alongside ROI3 and ROI8 in the standard grand-average report and the individual-alpha-peak-aligned report. For each of the four TFR measures (ON, OFF, baseline-corrected ON minus OFF, and the unbaselined symmetric ratio), ROI6 receives a separate plot and participant/channel inclusion documentation. The standard report also includes an ROI6 ERP average. ROI6 follows the same participant-weighting and bad-channel rules as the other ROIs. Peak-aligned ROI6 TFR figures are generated for each existing peak-aligned contrast.

## Unbaselined stimulation-OFF peak-aligned alpha by sensor

Run `G03_off_peak_aligned_alpha_sensor_curves.py` after generating the individual OFF-derived peak CSV and final cleaned no-stim cue epochs:

```bash
python analysis/group/EEG_all_channels/G03_off_peak_aligned_alpha_sensor_curves.py --subjects 101 102 103 104 105 106 107 108 110 111 112 113 114 115 116 118 119 120 121 122 123 --platform mac --n-jobs 1
```

For each participant, the script computes trial-averaged multitaper power from **unbaselined OFF epochs only**, then averages the bins within their individual alpha peak ±2 Hz. It plots the 0–1.2 s alpha-power time course for each good EEG sensor, averaging participants separately per sensor (equal subject weight; shaded between-subject SEM). The participant peak is read from `derivatives/reports/group/individual_alpha_peaks.csv`; use `--peaks-csv PATH` if stored elsewhere. There is **no baseline correction**, no stimulation ON analysis and no ROI averaging.

Outputs under `derivatives/reports/group/EEG_all_channels_OFF_peak_alpha/`:

- `off_peak_alpha_sensor_report.pdf`: ranking figure and individual sensor alpha curves.
- `sensor_alpha_power_ranking.png`: sensors ranked by absolute alpha power in 0.3–0.8 s, with a second bar for 0.1–0.7 s.
- `alpha_power_per_sensor.csv`: two absolute-power means per sensor, participant counts and included subject IDs.

Both window means are calculated from the group-mean alpha curve, and are expressed in original TFR power units (V²/Hz). The number of subjects can vary by sensor when channels are bad/missing. This is descriptive analysis; it does not test statistical significance.

## Q03: baseline-corrected stimulation-OFF alpha sensor curves

`Q03_off_peak_aligned_alpha_sensor_curves.py` reads cleaned OFF cue epochs and participant-specific OFF-derived peaks from `derivatives/reports/group/individual_alpha_peaks.csv`. It computes trial-averaged multitaper power, applies **percent baseline (-0.3 to -0.1 s) per participant/sensor/frequency**, then averages peak ±2 Hz and participants equally per sensor. Peaks in the CSV are not re-estimated. The 0–1.2 s curves and the two window means (0.3–0.8 and 0.1–0.7 s) are expressed in **percent change from baseline**.

Run from repository root:

```bash
python analysis/group/EEG_all_channels/Q03_off_peak_aligned_alpha_sensor_curves.py --subjects 101 102 103 104 105 106 107 108 110 111 112 113 114 115 116 118 119 120 121 122 123 --platform mac --n-jobs 1
```

Outputs: `derivatives/reports/group/EEG_all_channels_OFF_peak_alpha/` containing `off_peak_alpha_sensor_report.pdf`, `sensor_alpha_power_ranking.png`, and `alpha_power_per_sensor.csv`. The existing output directory name is retained; files are replaced when rerun.
