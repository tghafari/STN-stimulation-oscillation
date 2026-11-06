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
- Three-channel ROI: PO3, POz, PO4.
- Eight-channel ROI: PO3, POz, PO4, O1, Oz, O2, PO7, PO8.
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