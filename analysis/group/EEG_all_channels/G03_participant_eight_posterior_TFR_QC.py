#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""One PDF QC report containing all subjects, organised by TFR result.

For every result, each subject gets two plots only: mean of 8 posterior sensors
(PO3, POz, PO4, O1, Oz, O2, PO7, PO8) and mean of 3 sensors (PO3, POz, PO4).
There is no averaging across subjects.
"""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import matplotlib.pyplot as plt
import mne
import numpy as np
HERE=Path(__file__).resolve().parent;ANALYSIS_DIR=HERE.parents[1]
for p in (ANALYSIS_DIR/'subject'/'EEG_all_channels',ANALYSIS_DIR/'utils'):
 if str(p) not in sys.path:sys.path.insert(0,str(p))
from pipeline_config import CONDITIONS,resolve_project_root,stage_path
from pdf_report import ParticipantPDF
ROI8=('PO3','POz','PO4','O1','Oz','O2','PO7','PO8');ROI3=('PO3','POz','PO4');BASELINE=(-.3,-.1);FREQS=np.arange(2.,32.,.5);N_CYCLES=FREQS/2.;TIME_BANDWIDTH=2.;DECIM=2;TMIN=-.5;TMAX=1.5;ROBUST=98.
def parse_args():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--subjects',nargs='+',required=True);p.add_argument('--session',default='01');p.add_argument('--task',default='SpAtt');p.add_argument('--run',default='01');p.add_argument('--platform',choices=['mac','bluebear'],default='mac');p.add_argument('--project-root',default=None);p.add_argument('--n-jobs',type=int,default=4);return p.parse_args()
def load(root,s,a):
 out={}
 for c in CONDITIONS:
  f=stage_path(root,s,a.session,a.task,a.run,c,'clean','epo')
  if not f.exists():raise FileNotFoundError(f'Missing final cleaned epochs for sub-{s} {c}: {f}')
  e=mne.read_epochs(f,preload=True);k=[x for x in ('cue_onset_right','cue_onset_left') if x in e.event_id];out[c]=e[k] if k else e
 return out
def available(ep):
 st,no=ep['stim'],ep['no-stim'];eeg=st.copy().pick('eeg').ch_names;return[ch for ch in ROI8 if ch in eeg and ch in no.ch_names and ch not in st.info['bads'] and ch not in no.info['bads']]
def tfr(ep,picks,jobs):return ep.copy().pick(picks).compute_tfr(method='multitaper',freqs=FREQS,n_cycles=N_CYCLES,time_bandwidth=TIME_BANDWIDTH,use_fft=True,zero_mean=True,return_itc=False,average=True,decim=DECIM,n_jobs=jobs)
def ratio(st,no):
 x=st.copy();x.data=(st.data-no.data)/(st.data+no.data+np.finfo(float).eps);return x
def mean_roi(x,roi,name):
 picks=[ch for ch in roi if ch in x.ch_names]
 if not picks:return None,[]
 y=x.copy().pick(picks);y.data=y.data.mean(0,keepdims=True);y.info=mne.pick_info(y.info,[0],copy=True);y.info['chs'][0]['ch_name']=name;y.info['ch_names'][0]=name;return y,picks
def lim(*xs):
 vals=[]
 for x in xs:
  if x is None:continue
  ti=(x.times>=TMIN)&(x.times<=TMAX);z=np.asarray(x.data)[...,ti];z=z[np.isfinite(z)]
  if z.size:vals.append(z)
 if not vals:return(None,None)
 m=float(np.percentile(np.abs(np.concatenate(vals)),ROBUST));return(-m,m) if np.isfinite(m) and m>0 else(None,None)
def plot(x,title,v):
 kw=dict(picks=[x.ch_names[0]],tmin=TMIN,tmax=TMAX,fmin=2.,fmax=31.5,baseline=None,mode=None,show=False,colorbar=True,cmap='RdBu_r')
 if None not in v:kw['vlim']=v
 f=x.plot(**kw);f=f[0] if isinstance(f,list) else f;f.axes[0].axvline(0,color='k',ls='--',lw=.8);f.axes[0].set_title(title);return f
def main():
 a=parse_args();subjects=[s.removeprefix('sub-') for s in a.subjects];root=resolve_project_root(a.platform,a.project_root);out=root/'derivatives'/'reports'/'QC'/'posterior_TFR_all_subjects';figs=out/'figures';figs.mkdir(parents=True,exist_ok=True);report=ParticipantPDF(str(out),'posterior_TFR_QC_'+'_'.join(subjects));eps={s:load(root,s,a) for s in subjects};av={s:available(eps[s]) for s in subjects}
 report.add_text('Subjects and channel availability','Subjects: '+', '.join('sub-'+s for s in subjects)+'\n\n'+'\n'.join(f"sub-{s}: available={', '.join(av[s]) if av[s] else 'none'}; missing={', '.join(ch for ch in ROI8 if ch not in av[s]) or 'none'}" for s in subjects),'QC overview')
 report.add_text('Analysis details','One PDF contains all requested subjects. There is NO averaging across subjects. The report is organised by TFR result; within each result every subject is shown in sequence with two plots: within-subject mean of the 8-channel posterior ROI and within-subject mean of PO3/POz/PO4. Missing/rejected sensors are omitted, never interpolated. Multitaper TFR: 2-31.5 Hz in 0.5-Hz steps, n_cycles=f/2, time-bandwidth=2, FFT=True, zero_mean=True, ITC=False, trial-average=True, decim=2. Stim OFF/ON use percent baseline -0.3 to -0.1 s. Difference and normalized difference use original unbaselined power.','QC overview')
 raw={}
 for s in subjects:
  if not av[s]:continue
  print(f'Computing TFRs for sub-{s}');raw[s]={c:tfr(eps[s][c],av[s],a.n_jobs) for c in CONDITIONS}
 sections=[('Stim OFF','no-stim'),('Stim ON','stim'),('Stim ON - Stim OFF','difference'),('(Stim ON - Stim OFF) / (Stim ON + Stim OFF)','ratio')]
 for title,key in sections:
  report.add_text('Section details',f'{title}: all subjects are shown below in the requested order. For each subject: first mean of 8 posterior channels, then mean PO3/POz/PO4. No across-subject averaging.',title)
  for s in subjects:
   if s not in raw:
    report.add_text(f'sub-{s}',f'sub-{s}: no requested posterior sensors available.',title);continue
   st,no=raw[s]['stim'],raw[s]['no-stim']
   if key=='stim':x=st.copy().apply_baseline(BASELINE,mode='percent');cap='Percent baseline -0.3 to -0.1 s.'
   elif key=='no-stim':x=no.copy().apply_baseline(BASELINE,mode='percent');cap='Percent baseline -0.3 to -0.1 s.'
   elif key=='difference':x=st.copy();x.data=st.data-no.data;cap='Stim ON - Stim OFF from unbaselined power.'
   else:x=ratio(st,no);cap='(Stim ON - Stim OFF)/(Stim ON + Stim OFF) from unbaselined power.'
   m8,p8=mean_roi(x,ROI8,'Posterior_8_mean');m3,p3=mean_roi(x,ROI3,'Posterior_3_mean');v=lim(m8,m3);scale=f'Shared robust scale {v[0]:.4g} to {v[1]:.4g}.' if None not in v else 'Automatic scale.'
   report.add_figure(plot(m8,f'sub-{s}: {title} - mean 8 posterior channels',v),str(figs/f'{key}_sub-{s}_mean8.png'),f'sub-{s}: {title} - mean 8 channels',cap+' '+scale+f" Channels used: {', '.join(p8)}.",title)
   if m3 is not None:report.add_figure(plot(m3,f'sub-{s}: {title} - mean PO3/POz/PO4',v),str(figs/f'{key}_sub-{s}_mean3.png'),f'sub-{s}: {title} - mean PO3/POz/PO4',cap+' '+scale+f" Channels used: {', '.join(p3)}.",title)
   else:report.add_text(f'sub-{s}: 3-channel mean unavailable','None of PO3, POz, PO4 was available in both conditions.',title)
   plt.close('all')
 print(f'QC report complete: {report.pdf_fname}')
if __name__=='__main__':main()
