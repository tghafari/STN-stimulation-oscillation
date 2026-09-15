#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Single all-subject posterior TFR QC report with 4x4 subject grids.

For each TFR result the report contains TWO group-display figures, but NO group
averaging:
  1. a 4x4 grid showing the within-subject mean of PO3/POz/PO4 for each subject;
  2. a 4x4 grid showing the within-subject mean of the 8 posterior sensors
     PO3, POz, PO4, O1, Oz, O2, PO7, PO8 for each subject.

Each panel is one participant. With 16 subjects, every grid is exactly 4x4. If fewer
than 16 are supplied, unused panels are blank. More than 16 subjects are rejected so
that the QC layout remains explicitly 4x4.
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
ROI8=('PO3','POz','PO4','O1','Oz','O2','PO7','PO8');ROI3=('PO3','POz','PO4')
BASELINE=(-.3,-.1);FREQS=np.arange(2.,32.,.5);N_CYCLES=FREQS/2.;TIME_BANDWIDTH=2.;DECIM=2;TMIN=-.5;TMAX=1.5;ROBUST=98.
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
 st,no=ep['stim'],ep['no-stim'];eeg=st.copy().pick('eeg').ch_names
 return[ch for ch in ROI8 if ch in eeg and ch in no.ch_names and ch not in st.info['bads'] and ch not in no.info['bads']]
def compute(ep,picks,jobs):return ep.copy().pick(picks).compute_tfr(method='multitaper',freqs=FREQS,n_cycles=N_CYCLES,time_bandwidth=TIME_BANDWIDTH,use_fft=True,zero_mean=True,return_itc=False,average=True,decim=DECIM,n_jobs=jobs)
def ratio(st,no):
 x=st.copy();x.data=(st.data-no.data)/(st.data+no.data+np.finfo(float).eps);return x
def mean_roi(x,roi):
 picks=[ch for ch in roi if ch in x.ch_names]
 if not picks:return None,[]
 y=x.copy().pick(picks);y.data=y.data.mean(0,keepdims=True);return y,picks
def common_lim(subject_rois):
 vals=[]
 for x in subject_rois.values():
  if x is None:continue
  ti=(x.times>=TMIN)&(x.times<=TMAX);fi=(x.freqs>=2)&(x.freqs<=31.5);z=np.asarray(x.data)[:,fi][:,:,ti];z=z[np.isfinite(z)]
  if z.size:vals.append(z)
 if not vals:return(None,None)
 m=float(np.percentile(np.abs(np.concatenate(vals)),ROBUST));return(-m,m) if np.isfinite(m) and m>0 else(None,None)
def plot_grid(subjects,rois,title,v):
 fig,axes=plt.subplots(4,4,figsize=(20,18),constrained_layout=True);axes=axes.ravel();last_im=None
 for i,ax in enumerate(axes):
  if i>=len(subjects):ax.axis('off');continue
  s=subjects[i];x=rois.get(s)
  if x is None:ax.axis('off');ax.set_title(f'sub-{s}: unavailable');continue
  ti=(x.times>=TMIN)&(x.times<=TMAX);fi=(x.freqs>=2)&(x.freqs<=31.5);data=x.data[0][fi][:,ti];times=x.times[ti];freqs=x.freqs[fi]
  im=ax.pcolormesh(times,freqs,data,shading='auto',cmap='RdBu_r',vmin=v[0],vmax=v[1]);last_im=im;ax.axvline(0,color='k',ls='--',lw=.7);ax.set_title(f'sub-{s}',fontsize=11);ax.set_xlabel('Time (s)');ax.set_ylabel('Frequency (Hz)')
 if last_im is not None:fig.colorbar(last_im,ax=list(axes),shrink=.8,label='TFR power')
 fig.suptitle(title,fontsize=16);return fig
def main():
 a=parse_args();subjects=[s.removeprefix('sub-') for s in a.subjects]
 if len(subjects)>16:raise ValueError(f'This QC report uses a fixed 4x4 layout (maximum 16 subjects); received {len(subjects)}.')
 root=resolve_project_root(a.platform,a.project_root);out=root/'derivatives'/'reports'/'QC'/'posterior_TFR_all_subjects';figs=out/'figures';figs.mkdir(parents=True,exist_ok=True);report=ParticipantPDF(str(out),'posterior_TFR_QC_'+'_'.join(subjects));eps={s:load(root,s,a) for s in subjects};av={s:available(eps[s]) for s in subjects}
 report.add_text('Subjects and channel availability','Subjects: '+', '.join('sub-'+s for s in subjects)+'\n\n'+'\n'.join(f"sub-{s}: available={', '.join(av[s]) if av[s] else 'none'}; missing={', '.join(ch for ch in ROI8 if ch not in av[s]) or 'none'}" for s in subjects),'QC overview')
 report.add_text('Analysis details','One PDF contains all requested participants. No values are averaged across participants. For each TFR result there are two 4x4 figures: first the within-participant mean of PO3/POz/PO4 for all participants, then the within-participant mean of PO3/POz/PO4/O1/Oz/O2/PO7/PO8 for all participants. Each panel is one participant. Missing/rejected sensors are omitted from that participant ROI and never interpolated. Multitaper TFR: 2-31.5 Hz in 0.5-Hz steps, n_cycles=f/2, time-bandwidth=2, FFT=True, zero_mean=True, ITC=False, trial-average=True, decim=2. Stim OFF/ON use percent baseline -0.3 to -0.1 s. Difference and normalized difference use original unbaselined power. Each 4x4 figure uses one shared robust symmetric colour scale across subjects so participant magnitudes can be compared visually.','QC overview')
 raw={}
 for s in subjects:
  if not av[s]:continue
  print(f'Computing TFRs for sub-{s}');raw[s]={c:compute(eps[s][c],av[s],a.n_jobs) for c in CONDITIONS}
 sections=[('Stim OFF','no-stim'),('Stim ON','stim'),('Stim ON - Stim OFF','difference'),('(Stim ON - Stim OFF) / (Stim ON + Stim OFF)','ratio')]
 for title,key in sections:
  roi3={};roi8={};used3={};used8={}
  for s in subjects:
   if s not in raw:roi3[s]=None;roi8[s]=None;continue
   st,no=raw[s]['stim'],raw[s]['no-stim']
   if key=='stim':x=st.copy().apply_baseline(BASELINE,mode='percent');caption='Percent power change relative to -0.3 to -0.1 s baseline.'
   elif key=='no-stim':x=no.copy().apply_baseline(BASELINE,mode='percent');caption='Percent power change relative to -0.3 to -0.1 s baseline.'
   elif key=='difference':x=st.copy();x.data=st.data-no.data;caption='Stim ON minus Stim OFF from original unbaselined power.'
   else:x=ratio(st,no);caption='(Stim ON - Stim OFF)/(Stim ON + Stim OFF) from original unbaselined power.'
   roi3[s],used3[s]=mean_roi(x,ROI3);roi8[s],used8[s]=mean_roi(x,ROI8)
  v3=common_lim(roi3);v8=common_lim(roi8)
  report.add_text('Section details',f'{title}. Two subject-grid figures follow. First: mean PO3/POz/PO4. Second: mean 8 posterior channels. {caption} No across-subject averaging. Colour limits are shared across the 16 subject panels within each ROI figure.',title)
  report.add_figure(plot_grid(subjects,roi3,f'{title}: mean PO3 / POz / PO4 — subjects',v3),str(figs/f'{key}_all_subjects_mean3_4x4.png'),f'{title}: 3-channel posterior mean, all subjects (4 x 4)',caption+' Each panel is one participant; no across-participant averaging.',title)
  report.add_figure(plot_grid(subjects,roi8,f'{title}: mean 8 posterior channels — subjects',v8),str(figs/f'{key}_all_subjects_mean8_4x4.png'),f'{title}: 8-channel posterior mean, all subjects (4 x 4)',caption+' Each panel is one participant; no across-participant averaging.',title)
  report.add_text('Channels used per participant','3-channel ROI:\n'+'\n'.join(f"sub-{s}: {', '.join(used3.get(s,[])) or 'unavailable'}" for s in subjects)+'\n\n8-channel ROI:\n'+'\n'.join(f"sub-{s}: {', '.join(used8.get(s,[])) or 'unavailable'}" for s in subjects),title)
 print(f'QC report complete: {report.pdf_fname}')
if __name__=='__main__':main()
