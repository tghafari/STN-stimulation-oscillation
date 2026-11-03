# Group-level EEG analysis

This folder contains the posterior and all-channel group analyses. Individual
EEG preprocessing must be completed before group analyses.

## Posterior workflow

1. Run `analysis/group/EEG_posterior_channels/G03_individual_stimOn_stimOff_alpha_report.py` to establish individual alpha peaks where needed.
2. Run `G04_prepare_stimOnOff_LeftRight_block_alpha.py` for all intended participants. G04 is the **only supported interactive interface** for confirming stimulation order. Confirmations are persisted in `analysis/subject/stimulation_sequence_confirmed.json`.
3. Run `analysis/statistics/S02_absolute_alpha_block_permutation.py` for the primary two-sided whole-period permutation test.
4. Run `analysis/statistics/S02_sensitivity_alpha.py` to regenerate the full PDF with symmetric normalised sensitivity and leave-one-out checks.

## Output folders

### `derivatives/reports/group/EEG_posterior_channels_block_mapping/`

- `trial_alpha.csv` is the **one required machine-readable intermediate dataset**.
  Each row is one validated cleaned cue trial with absolute alpha power,
  task-block and stimulation-period identity, stimulation labels, alpha
  frequency peak and event-mapping information. S02 requires this dataset.
  Removing it would require recomputing alpha power or changing S02's
  input design; it cannot be replaced by a PDF.
- `G04_block_mapping_QC.pdf` summarises processed participants,
  sample counts, mapping coordinate systems, and failures.

G04 replaces its output on each run; for partial tests use `--output-dir`
so previous full-group data are protected.

### `derivatives/reports/group/absolute_alpha_block_permutation/`

- `absolute_alpha_block_permutation.pdf`: one combined report containing
  complete methods, absolute-power permutation null and observed effect,
  normalised sensitivity null and observed effect, the numerical leave-one-out
  results table, and participant exclusions with reasons.
- `figures/absolute_alpha_permutation.{svg,tiff}`:
  publication-quality primary distribution and observed effect.
- `figures/normalised_alpha_permutation.{svg,tiff}`:
  symmetric normalised distribution and observed effect.
- No separate null-distribution, participant-effect or leave-one-out CSVs
  are written by the updated scripts.

The figures mark the two-sided alpha 0.05 threshold computed from the
95th percentile of the *absolute* permutation statistics (shown at
plus/minus the threshold); the p-value is computed from the observed
absolute statistic using the plus-one Monte Carlo correction.

## Statistical definitions

**Primary:** For each participant, average power across their retained
ON trials, subtract the mean of their retained OFF trials, then take the
arithmetic mean of the participant differences. Each participant has two
ON and two OFF stimulation periods (each comprising two task blocks).
For each of 10,000 randomisations, reassign the ON/OFF labels of *whole
periods* within each participant, keeping two ON and two OFF. Trials never
move or get independently relabelled. The two-sided p-value compares the
observed absolute group difference with absolute null differences.

**Normalised sensitivity:** calculate for each participant
`(mean_ON - mean_OFF) / (mean_ON + mean_OFF)` and average across participants.
For every permutation, recalculate both ON and OFF means, including the
denominator. This reduces sensitivity to between-participant EEG power
scale, but can be unstable when combined ON+OFF power is tiny.

**Leave-one-out sensitivity:** Repeat the original absolute group-level
permutation test omitting one participant at a time. The report gives the
observed group difference and two-sided permutation p-value for each
omission. This is an influence diagnostic, not an exclusion rule.
No one-sided tests are performed.

## Current exclusions for the 16-person STN analysis

| Subject | Reason |
| --- | --- |
| 104 | VLM stimulation, not STN |
| 109 | No available data |
| 110 | VLM stimulation, not STN |
| 111 | Incomplete, six rather than eight task blocks |
| 117 | Substantial missing recording data |
| 119 | First DBS ON period: all 80 epochs removed during final cleaning |
| 122 | DBS ON cue-to-task-block mapping unresolved |

Document any later changes to eligibility and rerun the group report.

## Mapping and crop-time definitions

`analysis/subject/stimulation_cropped_time.json` defines crop times.
It is distinct from `stimulation_sequence_confirmed.json`.
The shared `A00_review_stimulation_blocks.py` and
`A02_validate_bids_epoch_mapping.py` modules under
`analysis/subject/EEG_STN_laterality/` perform block detection and
timing validation; **they do not provide a second supported confirmation
workflow**. `review_raw_stimulation_crops.py` is used only to inspect/change
crop intervals, *not* stimulation labels.
