#!/usr/bin/env python
"""Review actual BIDS block timing and confirm STN stimulation laterality.

The HTML/CSV review is derived from each participant's BIDS events.tsv. The proposed
sequence is only a proposal: the analysis runner requires explicit user confirmation.
"""
from __future__ import annotations
import argparse
import html
import json
import sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
SUBJECT_DIR = HERE.parent
PIPELINE_DIR = SUBJECT_DIR / "EEG_all_channels"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))
from pipeline_config import resolve_project_root

SEQUENCE_PATH = HERE / "stimulation_sequence_reference.json"

def events_path(root, subject, session, task, run):
    return (root / "data" / "BIDS" / f"sub-{subject}" / f"ses-{session}" / "eeg" /
            f"sub-{subject}_ses-{session}_task-{task}_run-{run}_events.tsv")

def norm_label(value):
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")

def get_blocks(events):
    if "trial_type" not in events or "onset" not in events:
        raise ValueError("BIDS events.tsv must contain onset and trial_type columns.")
    ev = events.copy()
    ev["onset"] = pd.to_numeric(ev["onset"], errors="coerce")
    ev = ev.dropna(subset=["onset"]).sort_values("onset").reset_index(drop=True)
    ev["_label"] = ev["trial_type"].map(norm_label)
    starts = ev[ev["_label"].str.contains("block_onset", regex=False)]
    ends = ev[ev["_label"].str.contains("block_end", regex=False)]
    blocks = []
    used_ends = set()
    for _, start_row in starts.iterrows():
        start = float(start_row["onset"])
        candidates = ends[(ends["onset"] > start) & (~ends.index.isin(used_ends))]
        if candidates.empty:
            continue
        later_starts = starts[starts["onset"] > start]["onset"]
        next_start = float(later_starts.min()) if len(later_starts) else float("inf")
        candidates = candidates[candidates["onset"] < next_start]
        if candidates.empty:
            continue
        end_idx = int(candidates.index[0])
        end = float(ends.loc[end_idx, "onset"])
        used_ends.add(end_idx)
        within = ev[(ev["onset"] >= start) & (ev["onset"] <= end)]
        labels = within["_label"].tolist()
        blocks.append({
            "onset_s": start, "offset_s": end, "duration_s": end - start,
            "cue_left_n": sum("cue_onset_left" in x for x in labels),
            "cue_right_n": sum("cue_onset_right" in x for x in labels),
            "n_events": int(len(within)),
            "event_types": "; ".join(sorted(set(within["trial_type"].astype(str)))),
        })
    return sorted(blocks, key=lambda x: x["onset_s"])

def read_reference():
    return json.loads(SEQUENCE_PATH.read_text(encoding="utf-8"))

def review_rows(root, subjects, session, task, run):
    reference = read_reference()
    output = []
    for raw_sub in subjects:
        sub = str(raw_sub).removeprefix("sub-")
        path = events_path(root, sub, session, task, run)
        if not path.exists():
            raise FileNotFoundError(f"Missing BIDS events file for sub-{sub}: {path}")
        events = pd.read_csv(path, sep="\t", na_values=["n/a", ""])
        blocks = get_blocks(events)
        proposed = reference.get(f"sub-{sub}", [])
        if len(blocks) != 4:
            note = (f"Found {len(blocks)} paired block intervals, expected 4. Do not guess; "
                    "resolve BIDS events or block boundaries.")
        else:
            note = ("Proposed chronology comes from the legacy stimulation sequence reference. "
                    "Manually verify against the BIDS times and source record.")
        if sub == "110":
            note += " Existing source code notes stimulation was on VLM, not STN; exclude from STN-specific analysis unless source records show otherwise."
        if len(proposed) != 4:
            note += " No four-block sequence proposal exists; assign conditions manually after review."
        for i, block in enumerate(blocks, start=1):
            proposal = proposed[i-1] if len(proposed) == 4 else "unknown"
            output.append({
                "subject": sub, "block_index_chronological": i, **block,
                "source_event_file": str(path),
                "wiki_proposed_condition": proposal,
                "confirmed_condition": "", "confirmed": "no", "review_note": note,
            })
        if not blocks:
            output.append({
                "subject": sub, "block_index_chronological": 0,
                "onset_s": "", "offset_s": "", "duration_s": "",
                "cue_left_n": "", "cue_right_n": "", "n_events": "", "event_types": "",
                "source_event_file": str(path), "wiki_proposed_condition": "unknown",
                "confirmed_condition": "", "confirmed": "no", "review_note": note,
            })
    return output

def numeric_key(value):
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value).strip()

def confirmation_key(row):
    return (str(row.get("subject", "")).strip(),
            str(row.get("block_index_chronological", "")).strip(),
            numeric_key(row.get("onset_s", "")),
            numeric_key(row.get("offset_s", "")))

def preserve_confirmations(rows, csv_path):
    if not csv_path.exists():
        return rows
    old = pd.read_csv(csv_path, dtype=str).fillna("")
    keep = {}
    for _, row in old.iterrows():
        keep[confirmation_key(row)] = (
            row.get("confirmed_condition", ""), row.get("confirmed", "no"))
    for row in rows:
        key = confirmation_key(row)
        if key in keep:
            row["confirmed_condition"], row["confirmed"] = keep[key]
    return rows

def write_html(rows, html_path, csv_path):
    cols = ["subject", "block_index_chronological", "onset_s", "offset_s",
            "duration_s", "cue_left_n", "cue_right_n", "n_events", "event_types",
            "wiki_proposed_condition", "confirmed_condition", "confirmed", "review_note"]
    html_rows = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(row.get(c, '')))}</td>" for c in cols)
        html_rows.append(f"<tr>{cells}</tr>")
    headers = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    document = f"""<!doctype html><html><head><meta charset="utf-8">
<title>STN block laterality confirmation</title>
<style>body{{font-family:Arial,sans-serif;margin:24px}}table{{border-collapse:collapse;font-size:12px}}
th,td{{border:1px solid #bbb;padding:6px;vertical-align:top}}th{{background:#eee;position:sticky;top:0}}
tr:nth-child(even){{background:#fafafa}}.warning{{color:#8b2600;font-weight:bold}}</style></head><body>
<h1>Confirm STN stimulation blocks against BIDS data</h1>
<p>Rows show block onset/end times, durations and left/right cue counts from each participant's BIDS events.tsv.
The proposed condition comes from a legacy stimulation sequence mapping and is not automatically accepted.</p>
<p class="warning">Review the source record. Fill confirmed_condition with exactly off, stim_left, stim_right, or exclude,
and set confirmed to yes only after checking the participant's block times. The runner refuses unconfirmed/ambiguous mappings.</p>
<p>CSV to edit: <code>{html.escape(str(csv_path))}</code></p>
<table><thead><tr>{headers}</tr></thead><tbody>{''.join(html_rows)}</tbody></table>
</body></html>"""
    html_path.write_text(document, encoding="utf-8")

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", nargs="+", required=True)
    p.add_argument("--session", default="01")
    p.add_argument("--task", default="SpAtt")
    p.add_argument("--run", default="01")
    p.add_argument("--platform", choices=["mac", "bluebear"], default="mac")
    p.add_argument("--project-root", default=None)
    args = p.parse_args()
    root = resolve_project_root(args.platform, args.project_root)
    out = root / "derivatives" / "reports" / "group" / "EEG_STN_laterality"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "confirm_stimulation_blocks.csv"
    html_path = out / "confirm_stimulation_blocks.html"
    rows = preserve_confirmations(
        review_rows(root, args.subjects, args.session, args.task, args.run), csv_path)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    write_html(rows, html_path, csv_path)
    print(f"\nBIDS block review written:\n  {html_path}\n  {csv_path}")
    print("Review the HTML, edit the CSV, and set confirmed=yes only after manual verification.")

if __name__ == "__main__":
    main()
