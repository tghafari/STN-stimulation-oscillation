# S02: posterior alpha stimulation statistics

`S02_absolute_alpha_block_permutation.py` is the single entry point for primary and sensitivity analyses.

**Prerequisites:** final cleaned cue epochs for ON and OFF, verified B03 `trial_alpha.csv` containing `trial_index_in_condition` and stimulation-period labels, and an OFF-derived participant alpha peak recorded in that CSV. The CSV's absolute `alpha_power` column is **not** the statistical input: the script recomputes trial-level multitaper TFRs, applies percent baseline (−0.3 to −0.1 s) independently to every trial/sensor, then averages individual alpha peak ±1 Hz over 0.3–0.8 s.

Run from repository root:

```bash
python analysis/statistics/S02_absolute_alpha_block_permutation.py --subjects 101 102 103 105 106 107 108 112 113 114 115 116 118 120 121 123 --platform mac
```

Options: `--input-csv PATH`, `--project-root PATH`, `--session`, `--task`, `--run`, `--n-jobs`, `--n-permutations` (default 10000), `--seed` (default 42).

**Outputs:** `<PROJECT_ROOT>/derivatives/reports/group/absolute_alpha_block_permutation/absolute_alpha_block_permutation.pdf` and SVG/TIFF permutation figures under `figures/`. The PDF begins with the included subject IDs, individual peak frequencies, sensors and input path, followed by all statistical results and participant effects.

**Tests:** two-sided within-participant whole-period permutation of the group mean ON−OFF (percentage points); exact two-sided sign test and paired Wilcoxon **signed-rank** test on participant differences; normalized sensitivity `(ON−OFF)/(|ON|+|OFF|)` with period-label permutations; and leave-one-participant-out primary permutations. ON/OFF means are pooled across retained trials within each participant, and participants contribute equally. The signed-rank test is used instead of the independent-sample rank-sum test because conditions are paired. The signed normalization differs from normalization of absolute power because percent changes can be negative.

**Caution:** the whole-period permutation test assumes stimulation labels are exchangeable under the null; stimulation order and carryover are not modeled. These tests should be interpreted together, not as independent confirmations.
