#!/usr/bin/env python
"""Interactive participant-by-participant STN stimulation confirmation and analysis.

Eight consecutive BIDS task blocks are grouped into four stimulation periods:
(1,2), (3,4), (5,6), (7,8). Each participant is confirmed in the terminal
and analysed immediately before moving on to the next participant.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from A00_review_stimulation_blocks import review_rows, preserve_confirmations, write_html
from A01_participant_laterality_analysis import analyze_subject
from pipeline_config import resolve_project_root

VALID = ("off", "stim_left", "stim_right")
PRETTY = {"off": "STN OFF", "stim_left": "LEFT STN ON",
          "stim_right": "RIGHT STN ON", "exclude": "EXCLUDE", "unknown": "UNKNOWN"}

def ask_choice(prompt, choices):
    while True:
        answer = input(prompt).strip().lower()
        if answer in choices:
            return answer
        print("Please enter one of: " + ", ".join(choices))

def proposed_order(rows):
    labels = rows["wiki_proposed_condition"].astype(str).str.lower().tolist()
    return labels if len(labels) == 4 else []

def valid_order(labels):
    return len(labels) == 4 and labels.count("off") == 2 and labels.count("stim_left") == 1 and labels.count("stim_right") == 1

def display_order(sub, rows, labels):
    print(f"\n{'='*64}\nsub-{sub} — proposed stimulation sequence")
    print("BIDS task blocks are paired: (1,2), (3,4), (5,6), (7,8).")
    for (_, row), label in zip(rows.iterrows(), labels):
        pair = row.get("source_task_blocks", "?")
        print(f"  Task blocks {pair:>4}: {PRETTY.get(label, label):<15} "
              f"BIDS {row['onset_s']}–{row['offset_s']} s, "
              f"cues L={row['cue_left_n']}, R={row['cue_right_n']}")
    if sub == "110":
        print("WARNING: legacy records indicate VLM stimulation, not STN. Verify before inclusion.")

def choose_order(sub, rows):
    labels = proposed_order(rows)
    if valid_order(labels):
        display_order(sub, rows, labels)
        answer = ask_choice("Is this stimulation order correct? [y/n/s(skip)/q(quit)]: ",
                            ("y", "n", "s", "q"))
        if answer in ("s", "q"):
            return answer, None
        if answer == "y":
            return "confirmed", labels
    else:
        print(f"\nsub-{sub}: no complete four-period proposal in the repository.")
        answer = ask_choice("Enter order manually [m], skip [s], or quit [q]? ",
                            ("m", "s", "q"))
        if answer != "m":
            return answer, None

    print("\nEnter the four stimulation periods in chronological order.")
    print("  l = LEFT STN ON, r = RIGHT STN ON, o = STN OFF")
    print("Example: o o l r")
    while True:
        response = input("Four labels [l/r/o] (or s to skip, q to quit): ").strip().lower()
        if response in ("s", "q"):
            return response, None
        parts = response.replace(",", " ").split()
        conversion = {"l": "stim_left", "r": "stim_right", "o": "off"}
        if len(parts) == 4 and all(x in conversion for x in parts):
            selected = [conversion[x] for x in parts]
            if valid_order(selected):
                display_order(sub, rows, selected)
                if ask_choice("Confirm this order? [y/n]: ", ("y", "n")) == "y":
                    return "confirmed", selected
        print("Need exactly two OFF periods, one LEFT STN ON and one RIGHT STN ON.")

def save_rows(rows, csv_path, html_path):
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    write_html(rows, html_path, csv_path)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", nargs="+", required=True)
    p.add_argument("--session", default="01")
    p.add_argument("--task", default="SpAtt")
    p.add_argument("--run", default="01")
    p.add_argument("--platform", choices=["mac", "bluebear"], default="mac")
    p.add_argument("--project-root", default=None)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--review-only", action="store_true",
                   help="Interactive confirmation only, no TFR calculations.")
    args = p.parse_args()
    subjects = [str(s).removeprefix("sub-") for s in args.subjects]
    root = resolve_project_root(args.platform, args.project_root)
    out = root / "derivatives" / "reports" / "group" / "EEG_STN_laterality"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "confirm_stimulation_blocks.csv"
    html_path = out / "confirm_stimulation_blocks.html"
    existing = pd.read_csv(csv_path, dtype=str).fillna("") if csv_path.exists() else pd.DataFrame()
    all_rows = existing.to_dict("records")
    outcomes = []

    for sub in subjects:
        print(f"\nReading BIDS blocks for sub-{sub} ...")
        try:
            fresh = review_rows(root, [sub], args.session, args.task, args.run)
            rows = preserve_confirmations(fresh, csv_path)
            df = pd.DataFrame(rows)
            if len(df) != 4:
                print(f"sub-{sub}: found {len(df)} condition-period rows, expected 4. "
                      "Cannot safely pair blocks; skipping this participant.")
                outcomes.append({"subject": f"sub-{sub}", "status": "needs BIDS review", "report": ""})
                continue
            df = df.sort_values("block_index_chronological").reset_index(drop=True)
            action, labels = choose_order(sub, df)
            if action == "q":
                print("Stopping at user request.")
                break
            if action == "s":
                outcomes.append({"subject": f"sub-{sub}", "status": "skipped", "report": ""})
                continue
            df["confirmed_condition"] = labels
            df["confirmed"] = "yes"
            all_rows = [row for row in all_rows if str(row.get("subject")) != sub]
            all_rows.extend(df.to_dict("records"))
            save_rows(all_rows, csv_path, html_path)
            print(f"Saved confirmed block labels for sub-{sub}.")
            if args.review_only:
                outcomes.append({"subject": f"sub-{sub}", "status": "confirmed only", "report": ""})
                continue
            try:
                print(f"Running TFR analysis for sub-{sub} ...")
                report = analyze_subject(root, sub, df, args)
                outcomes.append({"subject": f"sub-{sub}", "status": "success", "report": str(report)})
            except Exception as exc:
                print(f"FAILED sub-{sub}: {type(exc).__name__}: {exc}")
                outcomes.append({"subject": f"sub-{sub}", "status": "failed", "report": "", "error": str(exc)})
        except Exception as exc:
            print(f"Cannot review sub-{sub}: {type(exc).__name__}: {exc}")
            outcomes.append({"subject": f"sub-{sub}", "status": "review failed", "report": "", "error": str(exc)})
        finally:
            if outcomes:
                pd.DataFrame(outcomes).to_csv(out / "runner_summary.csv", index=False)
    print(f"\nFinished. Summary: {out / 'runner_summary.csv'}")
    print(f"Confirmed assignments: {csv_path}")

if __name__ == "__main__":
    main()
