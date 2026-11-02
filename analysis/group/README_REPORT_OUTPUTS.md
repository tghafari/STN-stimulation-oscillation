# Group EEG reports - file guide

This folder is generated under `<PROJECT_ROOT>/derivatives/reports/group/`.
It contains outputs from the STN-DBS posterior EEG pipeline.
Do not edit the generated CSVs by hand; regenerate from the corresponding
script when the analysis inputs change.

## EEG_posterior_channels_block_mapping/

Created by `analysis/group/EEG_posterior_channels/G04_prepare_block_alpha.py`.

| File | Contents | Used for |
| --- | --- | --- |
| `trial_alpha.csv` | One retained cue trial per row: subject, task block, stimulation period, ON/OFF and laterality, cue direction, BIDS cue onset, alpha peak and absolute alpha power (V squared). | Primary input to S02 and sensitivity analyses. |
| `block_summary.csv` | Per-block retained trial counts, average absolute alpha power, block boundaries and inferred-onset flags. | Quality control. |
| `mapping_audit.csv` | Timing-coordinate model, sample offsets, segment split (when applicable), mapped/retained counts and label provenance. | Mapping audit. |
| `G04_error_log.csv` | Skipped or failed participants, error stage, message and traceback. | Troubleshooting; not an exclusion list. |
| `diagnostics/*_timing_original.csv` | Nearest BIDS cue candidates and residuals under the original-coordinate model. | Debugging only; nearest candidates are not validated matches. |
| `diagnostics/*_timing_concatenated.csv` | Equivalent diagnostic under the concatenated-coordinate model. | Debugging only. |

G04 overwrites its output CSVs on each run. For partial tests use
`--output-dir` to protect the full-group data. The confirmed stimulation
sequence JSON is in `analysis/subject/stimulation_sequence_confirmed.json`.

## absolute_alpha_block_permutation/

Created by `analysis/statistics/S02_absolute_alpha_block_permutation.py`
and the optional companion scripts.

| File | Contents | Used for |
| --- | --- | --- |
| `participant_effects.csv` | Subject, alpha peak, retained ON/OFF trial counts, and participant mean ON minus OFF absolute alpha power (V squared). | Primary participant effects. |
| `permutation_null.csv` | One permuted group mean absolute ON minus OFF per iteration (V squared). | Primary two-sided permutation p-value and figures. |
| `absolute_alpha_block_permutation.pdf` | Primary analysis report with methods and plots. | Review and presentation. |
| `participant_percentage_effects.csv` | Optional subject-wise 100 * (ON minus OFF) / OFF mean. | Relative-scale sensitivity. |
| `permutation_null_percentage.csv` | Optional permutation distribution for the mean percentage change, with the OFF denominator recalculated for every shuffle. | Relative-scale sensitivity p-value and figures. |
| `leave_one_out_absolute.csv` | Optional omitted participant, remaining N, recalculated group absolute effect, and two-sided permutation p-value. | Influence check. |
| `absolute_alpha_permutation_figure.pdf/png/svg` | Absolute null histogram with the observed group mean marked. | Slides and publications. |
| `percentage_alpha_permutation_figure.pdf/png/svg` | Percentage null histogram with the observed group mean marked. | Slides and publications. |

Optional files only exist after the corresponding scripts are run.
The primary analysis uses 10,000 whole-stimulation-period permutations by
default, preserving two ON and two OFF periods per participant. Trials are
not independently shuffled.

## Why sensitivity tests?

**Percentage change** uses each participant's own OFF power as a reference.
It asks whether the group difference persists on a relative scale, rather
than being driven by participants with high absolute EEG power. Every
permutation must recalculate both ON and OFF means and the OFF denominator.
A very small OFF mean can make the percentage unstable.

**Leave-one-out** reruns the absolute-power permutation test after removing
one participant at a time. It shows whether the observed group effect and
p-value depend strongly on any one participant. It is not a method for
selecting exclusions, and a p-value crossing 0.05 does not by itself
invalidate the primary test.

The sensitivity scripts use two-sided tests only. These checks are
exploratory and do not replace the primary result.

## Reproducibility

Record the participant list, script version, random seed, permutation count,
G04 input CSV and exclusions alongside the reports. If you run G04 again,
the underlying trial table can change, so rerun S02 and its sensitivity
analyses to keep the reports consistent.
