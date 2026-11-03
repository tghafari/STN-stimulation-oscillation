# Posterior-channel group analysis

Run the subject-level posterior preprocessing before these group analyses.
See [the group README](../README.md) for G04/S02 instructions, report
outputs, exclusions, and permutation-test methods.

## Available report scripts

- `G01_concatenated_epochs_report.py`: concatenates trial epochs, then
  produces pooled posterior ERP/TFR plots; participants with more trials
  contribute more to pooled trial results.
- `G02_grand_average_report.py`: first computes participant ERP/TFR, then
  averages participants equally.
- `G03_individual_stimOn_stimOff_alpha_report.py`: participant-specific
  posterior TFR and alpha curves, individual alpha peak documentation.
- `B03_prepare_block_alpha.py`: prompts for stimulation
  order only when not already saved; validates BIDS trial-to-block mapping
  and writes the minimal `trial_alpha.csv` input plus a QC PDF.

The G01/G02 TFR displays may use percent baseline correction, whereas G04
and S02 analyse **absolute, non-baseline-corrected** alpha power during
the cue-locked attention interval. Do not confuse these measures.
