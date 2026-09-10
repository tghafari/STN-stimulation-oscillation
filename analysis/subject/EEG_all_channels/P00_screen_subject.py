"""Stage 0: human inclusion decision from the existing posterior-channel PDF.

The participant's established PDF report is opened and the researcher decides
whether the PO3/PO4/POz alpha modulation is sufficient for inclusion. The choice
is written both to JSON and to the persistent participant PDF report.
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from datetime import datetime
from pathlib import Path
from pipeline_config import POSTERIOR_SCREEN_CHANNELS, qc_dir, resolve_project_root
from all_channel_report import participant_report, fmt_channels


def parse_args():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--subject',required=True)
    p.add_argument('--platform',choices=['mac','bluebear'],default='mac'); p.add_argument('--project-root',default=None)
    p.add_argument('--decision',choices=['include','exclude'],default=None); return p.parse_args()


def find_reports(root:Path, subject:str):
    folder=root/'derivatives'/'reports'/f'sub-{subject}'; expected=folder/f'sub-{subject}_analysis_report.pdf'
    if expected.is_file(): return [expected]
    if not folder.is_dir(): return []
    files=list(folder.glob(f'sub-{subject}*.pdf')) or list(folder.glob('*.pdf'))
    return sorted(set(files),key=lambda p:p.stat().st_mtime,reverse=True)


def open_pdf(path:Path):
    path=path.resolve()
    if sys.platform=='darwin': subprocess.run(['open',str(path)],check=False)
    elif sys.platform.startswith('win'):
        import os; os.startfile(str(path))
    else: subprocess.run(['xdg-open',str(path)],check=False)


def main():
    a=parse_args(); subject=a.subject.removeprefix('sub-'); root=resolve_project_root(a.platform,a.project_root)
    reports=find_reports(root,subject); report_path=reports[0] if reports else None
    print('\n'+'='*72); print(f'SUBJECT SCREENING: sub-{subject}'); print('Inspect alpha modulation in: '+fmt_channels(POSTERIOR_SCREEN_CHANNELS)); print('='*72)
    if report_path: print(f'Opening participant PDF report:\n  {report_path}'); open_pdf(report_path)
    else: print('WARNING: participant PDF report not found automatically.')
    decision=a.decision
    while decision is None:
        ans=input('\nPosterior alpha modulation sufficient for inclusion? [y/n]: ').strip().lower()
        if ans in {'y','yes'}: decision='include'
        elif ans in {'n','no'}: decision='exclude'
    record={'subject':f'sub-{subject}','decision':decision,'criterion':'human inspection of alpha modulation in PO3, PO4, POz','screen_channels':list(POSTERIOR_SCREEN_CHANNELS),'report_opened':str(report_path) if report_path else None,'timestamp_local':datetime.now().astimezone().isoformat()}
    outfile=qc_dir(root,subject)/'inclusion_decision.json'; outfile.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    report=participant_report(root,subject)
    report.add_text('All-channel pipeline: inclusion screening',f"Analysis stage: P00 human inclusion screening\nChannels inspected: {fmt_channels(POSTERIOR_SCREEN_CHANNELS)}\nCriterion: visible posterior alpha modulation in the established posterior-channel analysis\nDecision: {decision.upper()}\nSource report opened: {report_path or 'not found automatically'}\nNo automatic numerical threshold was used.",'All-channel EEG preprocessing')
    print(f'Saved inclusion decision: {decision.upper()}\nUpdated PDF: {report.pdf_fname}')
if __name__=='__main__': main()
