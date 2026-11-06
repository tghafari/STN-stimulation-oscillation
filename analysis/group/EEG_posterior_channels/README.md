# Posterior-channel group analysis

Run the subject-level posterior preprocessing before these group analyses.
See [the group README](../README.md) for G04/S02 instructions, report
outputs, exclusions, and permutation-test methods.

## Available report scripts

- `G01_grand_average_report.py`: first computes participant ERP/TFR, then
  averages participants equally.
- `G02_individual_stimOn_stimOff_alpha_report.py`: participant-specific
  posterior TFR and alpha curves, individual alpha peak documentation.

The G01/G02 TFR displays may use percent baseline correction, whereas G04
and S02 analyse **absolute, non-baseline-corrected** alpha power during
the cue-locked attention interval. Do not confuse these measures.
