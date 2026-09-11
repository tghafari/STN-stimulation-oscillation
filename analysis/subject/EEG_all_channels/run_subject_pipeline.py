"""Semi-automatic all-channel EEG runner: preprocessing -> ERP -> TFR -> report summary."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
from pipeline_config import qc_dir,resolve_project_root
PREPROCESSING=HERE/'preprocessing'; SENSOR=HERE/'sensor'
STAGES=[(0,'posterior-alpha inclusion QC',PREPROCESSING/'P00_screen_subject.py'),(1,'LP100 + PyPREP/manual channel QC -> segmentation -> epochs -> reference',PREPROCESSING/'P01_epoch_pyprep_reref.py'),(2,'ICA on concatenated condition epochs',PREPROCESSING/'P02_ica.py'),(3,'manual epoch rejection + spectral QC',PREPROCESSING/'P03_manual_epoch_rejection.py'),(4,'ERP analysis',SENSOR/'A01_ERP.py'),(5,'TFR analysis',SENSOR/'A02_TFR.py'),(6,'manuscript-style analysis overview',SENSOR/'A03_analysis_overview.py')]
def parse_args():
 p=argparse.ArgumentParser(description=__doc__); g=p.add_mutually_exclusive_group(required=True); g.add_argument('--subjects',nargs='+'); g.add_argument('--range',nargs=2,type=int)
 p.add_argument('--session',default='01');p.add_argument('--task',default='SpAtt');p.add_argument('--run',default='01');p.add_argument('--platform',choices=['mac','bluebear'],default='mac');p.add_argument('--project-root',default=None);p.add_argument('--rereference',choices=['avg','none'],default='avg',help='EEG reference. Default: avg. Use --rereference none to keep the original reference.');p.add_argument('--from-stage',type=int,choices=range(7),default=0);p.add_argument('--continue-on-error',action='store_true');return p.parse_args()
def subjects(a):return [str(x).removeprefix('sub-') for x in a.subjects] if a.subjects else [str(x) for x in range(a.range[0],a.range[1]+1)]
def status(root,s):
 p=qc_dir(root,s)/'inclusion_decision.json';return json.loads(p.read_text()).get('decision') if p.exists() else None
def run_stage(script,s,a):
 cmd=[sys.executable,str(script),'--subject',s,'--platform',a.platform]
 if script.name!='A03_analysis_overview.py':cmd+=['--session',a.session,'--task',a.task,'--run',a.run]
 if script.name=='P01_epoch_pyprep_reref.py':cmd+=['--rereference',a.rereference]
 if a.project_root:cmd+=['--project-root',a.project_root]
 env=dict(__import__('os').environ);existing=env.get('PYTHONPATH','');env['PYTHONPATH']=str(HERE)+(__import__('os').pathsep+existing if existing else '')
 print('\nRUNNING: '+' '.join(cmd));subprocess.run(cmd,check=True,env=env)
def main():
 a=parse_args();root=resolve_project_root(a.platform,a.project_root);print(f'Platform: {a.platform} (default is mac)');print('Rereferencing:', 'average' if a.rereference=='avg' else 'none')
 for s in subjects(a):
  try:
   for n,label,script in STAGES:
    if n<a.from_stage:continue
    if n>0 and status(root,s)!='include':print(f'Stopping sub-{s}: inclusion status is {status(root,s)!r}.');break
    print(f'\nStage {n}: {label}');run_stage(script,s,a)
  except Exception as exc:
   print(f'ERROR sub-{s}: {type(exc).__name__}: {exc}')
   if not a.continue_on_error:raise
if __name__=='__main__':main()
