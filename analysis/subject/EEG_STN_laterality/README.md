# Shared stimulation mapping helpers (not an active laterality pipeline)

This directory retains utilities required by
`analysis/group/EEG_posterior_channels/G04_prepare_stimOnOff_LeftRight_block_alpha.py`.

- `A00_review_stimulation_blocks.py`: inspect BIDS task markers, construct
  eight task blocks (or report ambiguity), and derive a proposed
  four-period order from the original reference.
- `A02_validate_bids_epoch_mapping.py`: strictly map cleaned cue epochs
  to their original BIDS cue events, including validated crop-specific
  offsets when necessary.
- `review_raw_stimulation_crops.py`: optional raw-recording crop-time
  editor. It changes `stimulation_cropped_time.json`, **not** the
  stimulation order.

The interactive confirmation of stimulation ON/OFF/left/right labels is
implemented **only in G04**, which persists confirmations to
`analysis/subject/stimulation_sequence_confirmed.json`.
The former participant laterality runner and unfinished participant
laterality analysis were retired to eliminate the duplicate workflow.
Do not delete A00/A02: G04 requires both.
