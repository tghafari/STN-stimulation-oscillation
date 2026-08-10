# Automated single-subject EEG pipeline

This runner executes the five active subject-level scripts in order for one or more participants:

1. `preprocessing/P01_first_look_BIDS_conversion.py`
2. `preprocessing/P02_segmenting_stim.py`
3. `preprocessing/P03_epoching_SpAtt.py`
4. `sensor/A01_ERP.py`
5. `sensor/A02_three_channel_TFR.py`

The runner is intentionally **semi-automatic**. It automates the repetitive script-to-script progression, but it stops when human QC is required for stimulation segmentation or bad-epoch rejection.

## Files added for the automated workflow

- `run_subject_pipeline.py` — the command-line runner.
- `stimulation_cropped_time.json` — the persistent stimulation ON/OFF crop-time table.
- `README_AUTOMATED_PIPELINE.md` — these instructions.

The runner does **not** edit the five analysis scripts. It reads each script, substitutes the participant/path settings in memory, and executes the original analysis logic. This means you can continue testing or editing the individual scripts normally.

## Before you run it

### 1. Clone/update the repository

Run the pipeline from a local clone of this repository. The expected structure is:

```text
STN-stimulation-oscillation/
└── analysis/
    └── subject/
        ├── run_subject_pipeline.py
        ├── stimulation_cropped_time.json
        ├── preprocessing/
        │   ├── P01_first_look_BIDS_conversion.py
        │   ├── P02_segmenting_stim.py
        │   └── P03_epoching_SpAtt.py
        └── sensor/
            ├── A01_ERP.py
            └── A02_three_channel_TFR.py
```

note that un P01_first_look_BIDS_conversion.py you might need to change brainVision_basename 
according to the subject's file name.

### 2. Use the same Python environment that already runs the five scripts

The automated runner does not introduce a separate analysis environment. If you can run the five scripts individually, use that same environment.

At minimum, the current scripts/runner use packages including:

```text
mne
mne-bids
numpy
pandas
matplotlib
pyprep
```

Your repository PDF-report helper must also remain available at the path expected by the existing scripts.

### 3. Know your data project root

The runner needs the root directory of your **STN-in-PD data project**, not the GitHub repository. That directory is expected to contain at least:

```text
STN-in-PD/
├── data/
│   ├── data-organised/
│   └── BIDS/
└── derivatives/
```

You pass this path with `--project-root` every time you run the pipeline. This avoids hard-coding one person's computer path into the automated runner.

### 4. Use an interactive plotting backend

Two steps require GUI interaction:

- a raw-data browser when stimulation crop times are missing;
- the MNE epoch browser for manual bad-trial rejection.

Run the pipeline in an environment where MNE plots can open interactively. A normal local Python/IPython/terminal session on the analysis computer is appropriate. A non-interactive batch job is not appropriate for these manual QC steps.

## Basic usage

From the repository root:

```bash
python analysis/subject/run_subject_pipeline.py \
  --subjects 115 \
  --project-root "/path/to/STN-in-PD"
```

For several specific participants:

Note: not recommended, as brainvision basename for subjects might vary.

```bash
python analysis/subject/run_subject_pipeline.py \
  --subjects 115 116 118 121 \
  --project-root "/path/to/STN-in-PD"
```

For an inclusive numeric range, for example participants 115 through 123:

```bash
python analysis/subject/run_subject_pipeline.py \
  --range 115 123 \
  --project-root "/path/to/STN-in-PD"
```

The default BIDS settings are:

```text
session = 01
task    = SpAtt
run     = 01
```

Override them only when necessary:

```bash
python analysis/subject/run_subject_pipeline.py \
  --subjects 115 \
  --project-root "/path/to/STN-in-PD" \
  --session 01 \
  --task SpAtt \
  --run 01
```

## What happens for each subject

### Step 1 — P01: BIDS conversion

The runner executes `P01_first_look_BIDS_conversion.py` for the current participant. The participant number and project/repository paths are substituted in memory; the original file is not rewritten.

### Step 2 — stimulation crop times and P02

After P01, the runner checks:

```text
analysis/subject/stimulation_cropped_time.json
```

If both `no-stim` and `stim` entries already contain valid crop times, it prints:

```text
this subject has cropped times for stim on and stim off
```

and moves directly to P02 without opening the raw-data browser.

If either entry is missing or incomplete, the runner:

1. reads the participant's BIDS EEG file;
2. opens the raw recording interactively;
3. waits until you close the raw browser;
4. asks for the **NO-STIM** crop times;
5. asks for the **STIM** crop times;
6. validates the values;
7. saves them into `stimulation_cropped_time.json`;
8. runs `P02_segmenting_stim.py` using those saved values.

Crop times must be entered in seconds as either one retained interval:

```text
start end
```

or two retained intervals:

```text
start1 end1 start2 end2
```

Comma-separated values are also accepted. Examples:

```text
15 974
```

or

```text
878, 1289, 1870, 2280
```

Each start time must be smaller than its corresponding end time.

**Important:** after you add new crop times, `stimulation_cropped_time.json` changes in your local Git checkout. Commit that JSON file to GitHub so the next person/computer also knows the crop times and will not be asked again.

### Step 3 — P03: manual bad-trial rejection

The runner then executes `P03_epoching_SpAtt.py`.

The existing epoching script opens the MNE epoch browser with only these three posterior channels:

```text
PO3
PO4
POz
```

Use that browser to mark bad epochs manually. The rejected trial is removed from the full epoch object, not only from those three displayed channels.

When you are finished:

1. mark the bad trials in the MNE epoch browser;
2. close the browser;
3. P03 saves the cleaned epochs;
4. the runner automatically proceeds to ERP.

Do **not** close the terminal running the pipeline while the epoch browser is open.

### Step 4 — ERP

After P03 has saved the manually cleaned epochs, the runner executes:

```text
sensor/A01_ERP.py
```

No extra runner input is required unless the underlying ERP script itself raises an error.

### Step 5 — TFR

Finally, the runner executes:

```text
sensor/A02_three_channel_TFR.py
```

When it finishes, the runner moves to the next requested participant.

### adding note to the report

Use the snippet below anytime, to add text to the subject's report:

from pathlib import Path

project_root = Path(project_root)

report_folder = project_root / "derivatives" / "reports" / "sub-117"
report = ParticipantPDF(str(report_folder), "117")

report.add_text(
    "Subject notes",
    """This participant only shows very noisy data, which I assumed was stim on only, but in the PSD there is no 130 Hz peak. I think this participant should be excluded. Sirui does not have much information either.""",
    "Quality control",
)

## What happens if a subject fails?

By default, the pipeline stops immediately on the first exception. This is safer for analysis because a failed preprocessing step should not silently lead to later analyses based on missing or stale files.

If you deliberately want to continue with the remaining participants, add:

```bash
--continue-on-error
```

Example:

```bash
python analysis/subject/run_subject_pipeline.py \
  --range 115 123 \
  --project-root "/path/to/STN-in-PD" \
  --continue-on-error
```

At the end, the runner prints the subjects that failed and exits with a non-zero status.

## Recommended first test

Before running a large participant range, test one participant that you already know works line by line:

```bash
python analysis/subject/run_subject_pipeline.py \
  --subjects 115 \
  --project-root "/path/to/STN-in-PD"
```

Check that:

1. P01 creates/updates the expected BIDS data;
2. subject 115 reports that crop times already exist;
3. P02 creates the stim/no-stim segmented files;
4. the P03 browser shows only `PO3`, `PO4`, and `POz`;
5. bad trials remain rejected after the browser closes;
6. the cleaned epoch files are saved;
7. ERP completes;
8. TFR completes;
9. figures/reports appear in the same locations as when the scripts are run individually.

Only after that test should you launch a multi-subject range.

## Adding a new participant with no crop times

You do not need to edit Python code.

Run the new participant normally:

```bash
python analysis/subject/run_subject_pipeline.py \
  --subjects 124 \
  --project-root "/path/to/STN-in-PD"
```

After P01, the raw browser will open. Inspect the stimulation sequence, close the browser, enter the requested times, and the runner will save them to `stimulation_cropped_time.json` before continuing.

After the participant completes, review the JSON change and commit it to the repository.

## Troubleshooting

### `Required analysis script not found`

Make sure your checkout has the current `analysis/subject/preprocessing` and `analysis/subject/sensor` folder structure.

### Raw or epoch browser does not appear

You are probably using a non-interactive plotting backend/session. Run the pipeline locally in the same graphical environment in which `raw.plot()` and `epochs.plot()` already work for you.

### `Posterior QC channels missing`

The existing P03 script requires `PO3`, `PO4`, and `POz`. Check channel naming and any channel-removal steps before epoching.

### Crop times were entered incorrectly

Edit `analysis/subject/stimulation_cropped_time.json` carefully, or remove the subject entry and rerun that participant so the runner asks again. Each condition must contain either 2 or 4 numeric values.

### A later script uses the wrong participant

The runner patches the standard assignments (`subject`, `session`, `task`, `run`, `project_root`, and `GITHUB_ROOT`) in memory. If an analysis script is later refactored to use a different variable name or configuration system, update `_patch_script_source()` in `run_subject_pipeline.py` accordingly.

## Why the runner executes the original scripts instead of duplicating them

The five scripts remain the source of truth for the analysis. The runner only supplies runtime configuration, manages the persistent crop-time table, and controls the sequence. This reduces the risk that the automated version and the line-by-line version slowly become two different analyses.
