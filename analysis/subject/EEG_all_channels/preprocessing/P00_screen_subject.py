"""P00: human posterior-alpha inclusion QC.

This stage only opens the existing participant PDF and records the human
include/exclude decision based on PO3, PO4 and POz. Signal preprocessing starts
in P01. Keeping P00 separate makes the inclusion decision easy to audit.
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from datetime import datetime
from pathlib import Path
from pipeline_config import POSTERIOR_SCREEN_CHANNELS, qc_dir, resolve_project_root
from all_channel_report import participant_report, fmt_channels, remove_old_posterior_channel_quality_section

def parse_args():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--subject',required=True); p.add_argument('--session',default='01'); p.add_argument('--task',default='SpAtt'); p.add_argument('--run',default='01'); p.add_argument('--platform',choices=['mac','bluebear'],default='mac'); p.add_argument('--project-root',default=None); p.add_argument('--decision',choices=['include','exclude'],default=None); return p.parse_args()
def open_pdf(path:Path):
    if sys.platform=='darwin': subprocess.run(['open',str(path.resolve())],check=False)
    elif sys.platform.startswith('win'):
        import os; os.startfile(str(path.resolve()))
    else: subprocess.run(['xdg-open',str(path.resolve())],check=False)
def main():
    a=parse_args(); s=a.subject.removeprefix('sub-'); root=resolve_project_root(a.platform,a.project_root); report=participant_report(root,s)
    # Remove the old posterior-pipeline PyPREP/PSD section so channel-quality
    # results are not duplicated in the participant report.
    remove_old_posterior_channel_quality_section(report)
    if report.pdf_fname.exists(): print(f'Opening participant PDF:\n  {report.pdf_fname}'); open_pdf(report.pdf_fname)
    else: print(f'WARNING: participant PDF not found: {report.pdf_fname}')
    print('\nInspect posterior alpha modulation in '+fmt_channels(POSTERIOR_SCREEN_CHANNELS))
    decision=a.decision
    while decision is None:
        ans=input('Posterior alpha modulation sufficient for inclusion? [y/n]: ').strip().lower()
        if ans in {'y','yes'}: decision='include'
        elif ans in {'n','no'}: decision='exclude'
    rec={'subject':f'sub-{s}','decision':decision,'criterion':'human inspection of alpha modulation in PO3, PO4, POz','timestamp_local':datetime.now().astimezone().isoformat()}
    (qc_dir(root,s)/'inclusion_decision.json').write_text(json.dumps(rec,indent=2)+'\n',encoding='utf-8')
    print(f'Inclusion decision: {decision.upper()}')
if __name__=='__main__': main()
