# S02: unbaselined normalized alpha statistics

`S02_absolute_alpha_block_permutation.py` runs all statistical and sensitivity tests on the same participant-level normalized ratio, **(ON − OFF)/(ON + OFF)**, calculated entirely from unbaselined alpha power. No baseline correction is used in S02.

Prerequisites: final cleaned ON/OFF cue epochs; B03 verified `trial_alpha.csv` for stimulation-period assignments; `<PROJECT_ROOT>/derivatives/reports/group/individual_alpha_peaks.csv` for individual OFF-derived alpha frequencies. Power is averaged over available good posterior sensors, each participant's alpha peak ±1 Hz, and 0.3–0.8 s.

Run from repository root:

```bash
python analysis/statistics/S02_absolute_alpha_block_permutation.py --subjects 101 102 103 105 106 107 108 112 113 114 115 116 118 120 121 123 --platform mac
```

Primary test: upper-tail whole-stimulation-period permutation of mean squared participant ratios (either-direction magnitude). Sensitivity: mean absolute ratio permutation and leave-one-participant-out squared-ratio permutation. Secondary tests: two-sided signed-mean ratio permutation paired Wilcoxon signed-rank test of participant ratios against zero, and an exact two-sided sign test on positive versus negative participant ratios (zero ratios excluded). The signed-rank test assesses a consistent direction, not heterogeneous magnitude.

The PDF, permutation figures, and leave-one-out table are saved under `<PROJECT_ROOT>/derivatives/reports/group/absolute_alpha_block_permutation/`. The PDF includes participants, individual peaks, methods, test statistics, p-values and figure explanations. Optional `--input-csv`, `--peaks-csv`, `--project-root`, `--n-permutations`, `--seed`, and `--n-jobs` flags are supported.

Inference assumes whole-period stimulation labels are exchangeable; order/carryover effects are not modeled.

## S01: individual-peak cluster and whole-window tests

`S01_alpha_cluster_report.py` reads participant OFF-derived peaks from the same `individual_alpha_peaks.csv`, recomputes G02-compatible multitaper TFRs from final cleaned epochs, separately percent-baseline-corrects ON/OFF over -0.3 to -0.1 seconds, and tests ON - OFF against **zero** during 0.3-0.8 seconds. Alpha power is averaged over each participant's peak +/-1 Hz. A two-sided temporal cluster-permutation test is performed per sensor and ROI3/ROI8. Each ROI additionally reports a whole-window mean one-sample t-test against zero. Available good sensors are averaged within each participant before ROI testing. The optional `--sensor-fdr` applies BH-FDR across sensors using each sensor's minimum time-cluster-corrected p-value; sensors with no clusters enter with p=1. This does not correct the two ROI tests against one another.

```bash
python analysis/statistics/S01_alpha_cluster_report.py --subjects 101 102 103 105 106 107 108 112 113 114 115 116 118 120 121 123 --platform mac 
```

Output: `<PROJECT_ROOT>/derivatives/reports/group/EEG_all_channels_alpha_cluster/` (PDF, figures, sensor cluster CSV and analysis audit JSON). G02 does not persist TFR arrays, so S01 recomputes them from cleaned epochs. The tests are on participant-level baseline-percent differences, not on unbaselined ratios.

The primary squared and absolute magnitude tests use an upper-tail permutation null (large magnitudes are evidence against no effect). They do not establish the presence of true between-participant heterogeneity separately from measurement noise. The report includes null distributions for squared, absolute, signed, and unbaselined normalized effects, and a leave-one-out table of squared effects.
