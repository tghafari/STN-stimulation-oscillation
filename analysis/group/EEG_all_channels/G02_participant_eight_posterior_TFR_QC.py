#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Participant-level posterior TFR QC report: TWO ROI plots per TFR result.

There is NO averaging across participants. Each requested participant gets a separate
PDF quality-control report.

ROI8 = PO3, POz, PO4, O1, Oz, O2, PO7, PO8
ROI3 = PO3, POz, PO4

For EACH participant and EACH TFR result, exactly two TFR figures are produced:
  1. within-participant mean of the available ROI8 sensors
  2. within-participant mean of the available ROI3 sensors

TFR results, in report order:
  1. no stimulation
  2. stimulation
  3. stimulation - no stimulation
  4. (stimulation - no stimulation) / (stimulation + no stimulation)

Stim/no-stim are percent-baseline corrected (-0.3 to -0.1 s). The difference and
normalized difference are calculated from ORIGINAL UNBASELINED power and receive no
baseline correction. This keeps the QC contrasts directly interpretable and avoids
baseline normalization creating the contrast itself.

TFR settings match the posterior grand-average analysis:
2-31.5 Hz in 0.5-Hz steps; multitaper; n_cycles=f/2; time-bandwidth=2;
decim=2; FFT=True; zero_mean=True; ITC=False; trial-average=True.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import mne
import numpy as np
HERE=Path(__file__).resolve().parent; ANALYSIS_DIR=HERE.parents[1]
for p in (ANALYSIS_DIR/'subject'/'EEG_all_channels',ANALYSIS_DIR/'utils'):
 if str(p) not in sys.path:sys.path.insert(0,str(p))
from pipeline_config import CONDITIONS,resolve_project_root,stage_path
from pdf_report import ParticipantPDF
ROI8=('PO3','POz','PO4','O1','Oz','O2','PO7','PO8');ROI3=('PO3','POz','PO4')
BASELINE=(-.3,-.1);FREQS=np.arange(2.,32.,.5);N_CYCLES=FREQS/2.;TIME_BANDWIDTH=2.;DECIM=2;PLOT_TMIN=-.5;PLOT_TMAX=1.5;ROBUST=98.
def parse_args():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--subjects',nargs='+',required=True);p.add_argument('--session',default='01');p.add_argument('--task',default='SpAtt');p.add_argument('--run',default='01');p.add_argument('--platform',choices=['mac','bluebear'],default='mac');p.add_argument('--project-root',default=None);p.add_argument('--n-jobs',type=int,default=4);return p.parse_args()
def load_epochs(root,s,a):
 out={}
 for c in CONDITIONS:
  f=stage_path(root,s,a.session,a.task,a.run,c,'clean','epo')
  if not f.exists():raise FileNotFoundError(f'Missing final cleaned epochs for sub-{s} {c}: {f}')
  ep=mne.read_epochs(f,preload=True);cue=[x for x in ('cue_onset_right','cue_onset_left') if x in ep.event_id];out[c]=ep[cue] if cue else ep
 return out
def available_roi(pair,roi):
 st,no=pair['stim'],pair['no-stim'];eeg=st.copy().pick('eeg').ch_names
 return[ch for ch in roi if ch in eeg and ch in no.ch_names and ch not in st.info['bads'] and ch not in no.info['bads']]
def compute(ep,picks,jobs):return ep.copy().pick(picks).compute_tfr(method='multitaper',freqs=FREQS,n_cycles=N_CYCLES,time_bandwidth=TIME_BANDWIDTH,use_fft=True,zero_mean=True,return_itc=False,average=True,decim=DECIM,n_jobs=jobs)
def contrast(st,no,kind):
 x=st.copy()
 if kind=='difference':x.data=st.data-no.data
 elif kind=='ratio':x.data=(st.data-no.data)/(st.data+no.data+np.finfo(float).eps)
 else:raise ValueError(kind)
 return x
def roi_mean(tfr,channels,label):
 x=tfr.copy().pick(channels);x.data=x.data.mean(axis=0,keepdims=True);x.info=mne.create_info([label],x.info['sfreq'],'eeg');return x
def robust_lim(*tfrs):
 vals=[]
 for x in tfrs:
  ti=(x.times>=PLOT_TMIN)&(x.times<=PLOT_TMAX);fi=(x.freqs>=2)&(x.freqs<=31.5);z=np.asarray(x.data)[:,fi][:,:,ti];z=z[np.isfinite(z)]
  if z.size:vals.append(z)
 if not vals:return(None,None)
 m=float(np.percentile(np.abs(np.concatenate(vals)),ROBUST));return(-m,m) if np.isfinite(m) and m>0 else(None,None)
def plot_roi(x,label,title,lim):
 kw=dict(picks=[label],tmin=PLOT_TMIN,tmax=PLOT_TMAX,fmin=2,fmax=31.5,baseline=None,mode=None,show=False,colorbar=True,cmap='RdBu_r')
 if None not in lim:kw['vlim']=lim
 fig=x.plot(**kw);fig=fig[0] if isinstance(fig,list) else fig;fig.set_size_inches(9,6,forward=True);fig.axes[0].axvline(0,color='k',ls='--',lw=.8);fig.axes[0].set_title(title);return fig
def add_pair(report,figs,s,result8,result3,title,section,caption,av8,av3):
 # Shared scale between the 8-channel and 3-channel mean for the SAME participant/result,
 # making their visual magnitude directly comparable during QC.
 lim=robust_lim(result8,result3);scale=f'Shared robust symmetric scale: {lim[0]:.4g} to {lim[1]:.4g}.' if None not in lim else 'Automatic scale.'
 report.add_text('Analysis details',caption+' '+scale+f' ROI8 uses: {", ".join(av8)}. ROI3 uses: {", ".join(av3)}. Means are within this participant only; no across-subject averaging.',section)
 report.add_figure(plot_roi(result8,'ROI8_mean',f'sub-{s}: {title} — mean 8 posterior sensors',lim),str(figs/f'{section.replace(" ","_")}_mean8.png'),f'{title}: mean of 8 posterior sensors',caption+' '+scale,section)
 report.add_figure(plot_roi(result3,'ROI3_mean',f'sub-{s}: {title} — mean PO3/POz/PO4',lim),str(figs/f'{section.replace(" ","_")}_mean3.png'),f'{title}: mean PO3, POz and PO4',caption+' '+scale,section)
def build(root,s,a):
 ep=load_epochs(root,s,a);av8=available_roi(ep,ROI8);av3=available_roi(ep,ROI3)
 if not av8:raise RuntimeError(f'sub-{s}: no requested 8-channel posterior ROI sensors available in both conditions.')
 if not av3:raise RuntimeError(f'sub-{s}: none of PO3/POz/PO4 available in both conditions.')
 missing8=[ch for ch in ROI8 if ch not in av8];missing3=[ch for ch in ROI3 if ch not in av3]
 out=root/'derivatives'/'reports'/'QC'/'posterior_ROI_TFR'/f'sub-{s}';figs=out/'figures';figs.mkdir(parents=True,exist_ok=True);report=ParticipantPDF(str(out),f'{s}_posterior_ROI_TFR_QC')
 report.add_text('QC overview',f'Participant: sub-{s}\nNo across-participant averaging.\nStim OFF cue epochs: {len(ep["no-stim"])}\nStim ON cue epochs: {len(ep["stim"])}\nROI8 requested: {", ".join(ROI8)}\nROI8 available: {", ".join(av8)}\nROI8 unavailable/rejected: {", ".join(missing8) if missing8 else "none"}\nROI3 requested: {", ".join(ROI3)}\nROI3 available: {", ".join(av3)}\nROI3 unavailable/rejected: {", ".join(missing3) if missing3 else "none"}','QC summary')
 # Compute only the union needed by the two ROIs, once per condition.
 union=[ch for ch in ROI8 if ch in av8];raw={c:compute(ep[c],union,a.n_jobs) for c in CONDITIONS}
 no_disp=raw['no-stim'].copy().apply_baseline(BASELINE,mode='percent');st_disp=raw['stim'].copy().apply_baseline(BASELINE,mode='percent');diff=contrast(raw['stim'],raw['no-stim'],'difference');ratio=contrast(raw['stim'],raw['no-stim'],'ratio')
 results=[('1 No stimulation','No stimulation TFR',no_disp,f'Percent power change from baseline {BASELINE}.'),('2 Stimulation','Stimulation TFR',st_disp,f'Percent power change from baseline {BASELINE}.'),('3 Difference','Stimulation - no stimulation TFR',diff,'Stimulation minus no stimulation calculated from original unbaselined power; no baseline correction.'),('4 Normalized difference','(Stimulation - no stimulation) / (stimulation + no stimulation)',ratio,'Normalized contrast calculated from original unbaselined power; no baseline correction.')]
 for section,title,x,caption in results:
  r8=roi_mean(x,av8,'ROI8_mean');r3=roi_mean(x,av3,'ROI3_mean');add_pair(report,figs,s,r8,r3,title,section,caption,av8,av3)
 report.add_text('QC analysis parameters',f'Cue-locked final cleaned epochs only; cue-left/right combined within stimulation condition.\nTFR: multitaper; 2-31.5 Hz in 0.5-Hz steps; n_cycles=f/2; time-bandwidth={TIME_BANDWIDTH:g}; FFT=True; zero_mean=True; ITC=False; trial-average=True; decim={DECIM}.\nCondition baseline: {BASELINE}, percent.\nDifference: raw stim - raw no-stim.\nRatio: (raw stim - raw no-stim)/(raw stim + raw no-stim).\nEach ROI mean is calculated within sub-{s} only. No participant grand average is calculated.','QC summary')
 audit={'subject':f'sub-{s}','purpose':'participant-level posterior TFR QC','no_across_subject_average':True,'ROI8_requested':ROI8,'ROI8_available':av8,'ROI3_requested':ROI3,'ROI3_available':av3,'n_epochs':{'no-stim':len(ep['no-stim']),'stim':len(ep['stim'])},'tfr':{'freq_min_hz':2.0,'freq_max_hz':31.5,'step_hz':0.5,'n_cycles':'frequency/2','time_bandwidth':2.0,'decim':2,'condition_baseline':BASELINE,'difference_baseline':None,'ratio_baseline':None}}
 (out/f'sub-{s}_posterior_ROI_TFR_QC.json').write_text(json.dumps(audit,indent=2)+'\n');print(f'sub-{s}: {report.pdf_fname}')
def main():
 a=parse_args();root=resolve_project_root(a.platform,a.project_root)
 for s in [x.removeprefix('sub-') for x in a.subjects]:build(root,s,a)
if __name__=='__main__':main()
