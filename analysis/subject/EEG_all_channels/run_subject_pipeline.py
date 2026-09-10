"""Semi-automatic all-channel EEG runner: preprocessing -> ERP -> TFR -> report summary."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
from pipeline_config import qc_dir, resolve_project_root

HERE=Path(__file__).resolve().parent
STAGES=[
    (0,'posterior-alpha inclusion QC',HERE/'P00_screen_subject.py'),
    (1,'full continuous LP100 + PyPREP -> stim segmentation -> epochs -> rereference',HERE/'P01_epoch_pyprep_reref.py'),
    (2,'ICA fit on concatenated rereferenced stim/no-stim epochs and applied separately',HERE/'P02_ica.py'),
    (3,'manual rejection of ICA-cleaned stim/no-stim epochs + final spectral QC',HERE/'P03_manual_epoch_rejection.py'),
    (4,'ERP: all common good sensors and PO3/PO4/POz',HERE/'A01_ERP.py'),
    (5,'TFR: condition maps, stim-no-stim difference and normalized difference',HERE/'A02_TFR.py'),
    (6,'manuscript-style participant analysis overview',HERE/'A03_analysis_overview.py'),
]

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--subjects',nargs='+')
    g.add_argument('--range',nargs=2,type=int)
    p.add_argument('--session',default='01'); p.add_argument('--task',default='SpAtt'); p.add_argument('--run',default='01')
    p.add_argument('--platform',choices=['mac','bluebear'],default='mac'); p.add_argument('--project-root',default=None)
    p.add_argument('--from-stage',type=int,choices=range(7),default=0); p.add_argument('--continue-on-error',action='store_true')
    return p.parse_args()

def subjects(a):
    return [str(x).removeprefix('sub-') for x in a.subjects] if a.subjects else [str(x) for x in range(a.range[0],a.range[1]+1)]

def status(root,s):
    p=qc_dir(root,s)/'inclusion_decision.json'
    return json.loads(p.read_text()).get('decision') if p.exists() else None

def run_stage(script,s,a):
    cmd=[sys.executable,str(script),'--subject',s,'--platform',a.platform]
    if script.name != 'A03_analysis_overview.py':
        cmd += ['--session',a.session,'--task',a.task,'--run',a.run]
    if a.project_root: cmd += ['--project-root',a.project_root]
    print('\nRUNNING: '+' '.join(cmd)); subprocess.run(cmd,check=True)

def main():
    a=parse_args(); root=resolve_project_root(a.platform,a.project_root)
    for s in subjects(a):
        try:
            for n,label,script in STAGES:
                if n<a.from_stage: continue
                if n>0 and status(root,s)!='include':
                    print(f'Stopping sub-{s}: inclusion status is {status(root,s)!r}.'); break
                print(f'\nStage {n}: {label}'); run_stage(script,s,a)
        except Exception as exc:
            print(f'ERROR sub-{s}: {type(exc).__name__}: {exc}')
            if not a.continue_on_error: raise

if __name__=='__main__': main()
