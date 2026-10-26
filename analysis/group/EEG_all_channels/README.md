# All-channel EEG group analysis

## Overview

This folder contains the group-level ERP, TFR, alpha-power quality-control, and cluster-permutation analyses for the STN stimulation study.

All scripts use the final cleaned EEG epochs from each participant, separately for stimulation ON and OFF. Attention-left and attention-right cue epochs are combined.

## Scripts

| Script | Description |
|---|---|
| `G01_complete_grand_average_report.py` | Main group-analysis implementation. Calculates grand-average ERP and TFR, sensor-level stimulation contrasts, and posterior ROI summaries. |
| `run_grand_average.py` | Recommended entry point for G01. Adds detailed scalp and posterior figures to the group PDF. |
| `Q01_participant_eight_posterior_TFR_QC.py` | Displays individual-participant TFRs in 4 × 4 grids for three and eight posterior channels, separately for stimulation ON, OFF, difference, and normalized difference. |
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

Creates participant grids for three posterior channels (PO3, POz, PO4) and eight posterior channels (PO3, POz, PO4, O1, Oz, O2, PO7, PO8). No averaging across participants.

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
- TFR differences and normalized differences: original unbaselined power.
- Three-channel ROI: PO3, POz, PO4.
- Eight-channel ROI: PO3, POz, PO4, O1, Oz, O2, PO7, PO8.
- Bad or missing channels are not interpolated for group analysis.
- The subject is the unit of statistical inference.

## Outputs

- Main group PDF: `derivatives/reports/group/EEG_all_channels_complete_grand_average/`
- Q01 PDF: `derivatives/reports/QC/posterior_TFR_all_subjects/`
- Q02 figures: `derivatives/reports/group/EEG_all_channels_alpha_QC/`
- S01 statistical PDF: `derivatives/reports/group/EEG_all_channels_alpha_cluster/`

## Recommended order

Run `Q01` first to inspect individual-participant TFRs, followed by `Q02` for descriptive alpha-power QC. Run `run_grand_average.py` for the main group figures and `S01` for statistical inference.

These scripts should not be deleted: they perform different, complementary analyses.