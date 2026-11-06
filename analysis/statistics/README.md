# S02: posterior alpha stimulation statistics

`S02_absolute_alpha_block_permutation.py` is the single entry point for primary and sensitivity analyses.

**Prerequisites:** final cleaned cue epochs for ON and OFF, verified B03 `trial_alpha.csv` containing `trial_index_in_condition` and stimulation-period labels, and a participant-specific OFF-derived alpha peak from `<PROJECT_ROOT>/derivatives/reports/group/individual_alpha_peaks.csv`. The individual peak CSV is authoritative for frequency selection; B03 supplies verified trial-period mapping. The CSV's absolute `alpha_power` column is **not** the statistical input: the script recomputes trial-level multitaper TFRs, applies percent baseline (−0.3 to −0.1 s) independently to every trial/sensor, then averages individual alpha peak ±1 Hz over 0.3–0.8 s.

Run from repository root:

```bash
python analysis/statistics/S02_absolute_alpha_block_permutation.py --subjects 101 102 103 105 106 107 108 112 113 114 115 116 118 120 121 123 --platform mac
```

Options: `--peaks-csv PATH`, `--input-csv PATH`, `--project-root PATH`, `--session`, `--task`, `--run`, `--n-jobs`, `--n-permutations` (default 10000), `--seed` (default 42).

**Outputs:** `<PROJECT_ROOT>/derivatives/reports/group/absolute_alpha_block_permutation/absolute_alpha_block_permutation.pdf` and SVG/TIFF permutation figures under `figures/`. The PDF begins with the included subject IDs, individual peak frequencies, sensors and input path, followed by all statistical results and participant effects, a leave-one-out results table, and plots of the sign and paired signed-rank tests.

**Tests:** **primary** upper-tail whole-period permutation of the mean squared participant baseline-corrected ON−OFF difference (squared percentage points), detecting either-direction effects; **sensitivity** upper-tail whole-period permutation of mean absolute baseline-corrected differences; leave-one-out repeats of the squared primary statistic; **secondary** two-sided within-participant whole-period permutation of the signed group mean ON−OFF (percentage points); exact two-sided sign test and paired Wilcoxon **signed-rank** test on participant differences; unbaselined normalized sensitivity `(ON−OFF)/(ON+OFF)` with period-label permutations; and leave-one-participant-out primary permutations. ON/OFF means are pooled across retained trials within each participant, and participants contribute equally. The signed-rank test is used instead of the independent-sample rank-sum test because conditions are paired. The normalized ratio is computed exclusively from unbaselined absolute power, not percent-corrected power.

**Caution:** the whole-period permutation test assumes stimulation labels are exchangeable under the null; stimulation order and carryover are not modeled. These tests should be interpreted together, not as independent confirmations.

## S01: individual-peak cluster and whole-window tests

`S01_alpha_cluster_report.py` reads participant OFF-derived peaks from the same `individual_alpha_peaks.csv`, recomputes G02-compatible multitaper TFRs from final cleaned epochs, separately percent-baseline-corrects ON/OFF over -0.3 to -0.1 seconds, and tests ON - OFF against **zero** during 0.3-0.8 seconds. Alpha power is averaged over each participant's peak +/-1 Hz. A two-sided temporal cluster-permutation test is performed per sensor and ROI3/ROI8. Each ROI additionally reports a whole-window mean one-sample t-test against zero. Available good sensors are averaged within each participant before ROI testing. The optional `--sensor-fdr` applies BH-FDR across sensors using each sensor's minimum time-cluster-corrected p-value; sensors with no clusters enter with p=1. This does not correct the two ROI tests against one another.

```bash
python analysis/statistics/S01_alpha_cluster_report.py --subjects 101 102 103 105 106 107 108 112 113 114 115 116 118 120 121 123 --platform mac --peaks-csv /path/to/individual_alpha_peaks.csv
```

Output: `<PROJECT_ROOT>/derivatives/reports/group/EEG_all_channels_alpha_cluster/` (PDF, figures, sensor cluster CSV and analysis audit JSON). G02 does not persist TFR arrays, so S01 recomputes them from cleaned epochs. The tests are on participant-level baseline-percent differences, not on unbaselined ratios.

The primary squared and absolute magnitude tests use an upper-tail permutation null (large magnitudes are evidence against no effect). They do not establish the presence of true between-participant heterogeneity separately from measurement noise. The report includes null distributions for squared, absolute, signed, and unbaselined normalized effects, and a leave-one-out table of squared effects.
