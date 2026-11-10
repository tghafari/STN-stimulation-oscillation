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

## S01: exploratory alpha/theta targeted frequency-time clusters

`S01_alpha_theta_targeted_clusters.py` replaces the previous broad
`S01_alpha_cluster_report.py` workflow. It tests two **post-hoc,
visually selected** frequency-time windows, separately for each usable
single EEG sensor, posterior ROI3 (PO3, POz, PO4), and posterior ROI8
(PO3, POz, PO4, O1, Oz, O2, PO7, PO8).

| Band | Frequency axis | Cue-relative interval |
| --- | --- | --- |
| Alpha | Individual OFF-derived alpha peak -2.5 to +1 Hz | 0.1-0.7 s |
| Theta | Absolute 4-5.5 Hz | 0.1-0.4 s |

The input is cleaned ON and OFF cue epochs. Multitaper power is
trial-averaged in each condition, without baseline correction, and
the dimensionless contrast `(ON-OFF)/(ON+OFF)` is computed for each
participant. ROI sensors are averaged **within each participant and
condition before normalisation**. Alpha maps are interpolated to a
common peak-relative 0.5-Hz grid; theta uses absolute frequencies.

Each sensor/ROI and band is tested using a two-sided, one-sample
frequency-by-time cluster-mass sign-flip permutation test, with
cluster-level correction across the selected window. Across all
sensor/ROI x band tests, BH-FDR is additionally applied to each
test's minimum cluster p-value (or 1 when no clusters exist).
A PDF provides the methods, per-test participant counts, p-values,
cluster details and grand-average normalised maps with black outlines
around clusters with within-test corrected p <= 0.05.

**Important:** These windows were chosen from observed grand averages.
The resulting p-values are exploratory and do not provide independent
confirmatory evidence for those patterns. To make confirmatory claims,
preregister the windows and analyse an independent dataset.

```bash
python analysis/statistics/S01_alpha_theta_targeted_clusters.py \
  --subjects 101 102 103 105 106 107 108 112 113 114 115 116 118 120 121 123
```

Output: `<PROJECT_ROOT>/derivatives/reports/group/EEG_posterior_alpha_theta_clusters/S01_alpha_theta_targeted_clusters.pdf`.
No additional CSV is written by the new S01.

The primary squared and absolute magnitude tests use an upper-tail permutation null (large magnitudes are evidence against no effect). They do not establish the presence of true between-participant heterogeneity separately from measurement noise. The report includes null distributions for squared, absolute, signed, and unbaselined normalized effects, and a leave-one-out table of squared effects.
