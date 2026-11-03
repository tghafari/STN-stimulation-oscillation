# Block preparation for posterior STN-DBS EEG

This directory contains the single supported stimulation-sequence
confirmation and trial-to-block preparation workflow.

- `B01_review_bids_blocks.py`: identify BIDS task-block intervals and
  propose four stimulation periods; supporting module.
- `B02_validate_epoch_mapping.py`: verify cleaned EEG cue epochs against
  original BIDS events with strict timing checks; supporting module.
- `B03_prepare_block_alpha.py`: **the script to run**. Reuses previously
  confirmed stimulation orders, prompts only for missing orders (or when
  `--reconfirm` is supplied), calculates unbaselined trial-level posterior
  alpha power, and produces the validated input and QC PDF.

Run from repository root:

```bash
python analysis/subject/EEG_posterior_channels/block_preparation/B03_prepare_block_alpha.py \
    --subjects 101 102 103 105 106 107 108 112 113 114 115 116 118 120 121 123
```

Confirmed stimulation orders remain in
`analysis/subject/stimulation_sequence_confirmed.json`, not here.
Crop times remain in `analysis/subject/stimulation_cropped_time.json`.
These are different concepts: stimulation order vs recording crop intervals.

B03 writes `trial_alpha.csv` (required input for S02) and
`G04_block_mapping_QC.pdf` in the configured group report directory.
Existing report outputs are not moved or deleted by reorganising scripts.

The former EEG_STN_laterality folder is not needed for this active
block-preparation workflow. Laterality analysis can be implemented
separately in the future.
