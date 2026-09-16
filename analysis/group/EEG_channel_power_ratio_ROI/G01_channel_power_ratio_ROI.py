#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Independent alternative posterior ROI analysis: channel power -> channel ratio -> ROI ratio -> group mean.

This folder is deliberately separate from EEG_all_channels and does not import or
modify its group-analysis scripts.

Order of operations
-------------------
For EACH participant:
  1. Compute cue-locked TFR power separately for every available EEG channel for
     stimulation and no stimulation.
  2. For every channel calculate, from original unbaselined power:
         R_ch = (P_stim,ch - P_no-stim,ch) / (P_stim,ch + P_no-stim,ch)
  3. ROI3 participant result = arithmetic mean of available channel ratios for
     PO3, POz, PO4.
  4. ROI8 participant result = arithmetic mean of available channel ratios for
     PO3, POz, PO4, O1, Oz, O2, PO7, PO8.

GROUP:
  5. Grand ROI3 ratio = arithmetic mean of participant ROI3 ratios.
  6. Grand ROI8 ratio = arithmetic mean of participant ROI8 ratios.

Thus this is NOT EEG averaging before TFR, NOT ratio of group powers, and NOT
channel-power averaging before the ratio.

Missing/rejected sensors are not interpolated. A participant contributes to an ROI
when at least one requested ROI sensor is retained in BOTH conditions; the exact
channels used are written to the report and JSON metadata.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import matplotlib.pyplot as plt
import mne
import numpy as np
HERE=Path(__file__).resolve().parent
ANALYSIS_DIR=HERE.parents[1]
for p in (ANALYSIS_DIR/'subject'/'EEG_all_channels',ANALYSIS_DIR/'utils'):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
from pipeline_config import CONDITIONS,resolve_project_root,stage_path
from pdf_report import ParticipantPDF
ROI3=('PO3','POz','PO4')
ROI8=('PO3','POz','PO4','O1','Oz','O2','PO7','PO8')
FREQS=np.arange(2.,32.,.5);N_CYCLES=FREQS/2.;TIME_BANDWIDTH=2.;DECIM=2
TMIN=-.5;TMAX=1.5;ROBUST=98.
def parse_args():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--subjects',nargs='+',required=True);p.add_argument('--session',default='01');p.add_argument('--task',default='SpAtt');p.add_argument('--run',default='01');p.add_argument('--platform',choices=['mac','bluebear'],default='mac');p.add_argument('--project-root',default=None);p.add_argument('--n-jobs',type=int,default=4);return p.parse_args()
def load(root,s,a):
    out={}
    for c in CONDITIONS:
        f=stage_path(root,s,a.session,a.task,a.run,c,'clean','epo')
        if not f.exists():raise FileNotFoundError(f'Missing cleaned epochs sub-{s} {c}: {f}')
        e=mne.read_epochs(f,preload=True);k=[x for x in ('cue_onset_right','cue_onset_left') if x in e.event_id];out[c]=e[k] if k else e
    return out
def common_roi_channels(ep):
    st,no=ep['stim'],ep['no-stim'];eeg=st.copy().pick('eeg').ch_names
    return[ch for ch in ROI8 if ch in eeg and ch in no.ch_names and ch not in st.info['bads'] and ch not in no.info['bads']]
def tfr(ep,picks,jobs):
    return ep.copy().pick(picks).compute_tfr(method='multitaper',freqs=FREQS,n_cycles=N_CYCLES,time_bandwidth=TIME_BANDWIDTH,use_fft=True,zero_mean=True,return_itc=False,average=True,decim=DECIM,n_jobs=jobs)
def channel_ratio(st,no):
    x=st.copy();x.data=(st.data-no.data)/(st.data+no.data+np.finfo(float).eps);return x
def participant_roi_ratio(ratio,roi):
    picks=[ch for ch in roi if ch in ratio.ch_names]
    if not picks:return None,[]
    x=ratio.copy().pick(picks);x.data=x.data.mean(axis=0,keepdims=True);x.info=mne.pick_info(x.info,[0],copy=True);x.info['chs'][0]['ch_name']='ROI_ratio';x.info['ch_names'][0]='ROI_ratio';return x,picks
def grand_mean(xs):
    if not xs:return None
    g=xs[0].copy();g.data=np.mean([x.data for x in xs],axis=0);g.nave=len(xs);return g
def vlim(x):
    ti=(x.times>=TMIN)&(x.times<=TMAX);z=np.asarray(x.data)[...,ti];z=z[np.isfinite(z)]
    if not z.size:return(None,None)
    m=float(np.percentile(np.abs(z),ROBUST));return(-m,m) if np.isfinite(m) and m>0 else(None,None)
def plot(x,title):
    v=vlim(x);kw=dict(picks=['ROI_ratio'],tmin=TMIN,tmax=TMAX,fmin=2,fmax=31.5,baseline=None,mode=None,show=False,colorbar=True,cmap='RdBu_r')
    if None not in v:kw['vlim']=v
    f=x.plot(**kw);f=f[0] if isinstance(f,list) else f;f.axes[0].axvline(0,color='k',ls='--',lw=.8);f.axes[0].set_title(title);return f,v
def main():
    a=parse_args();subjects=[s.removeprefix('sub-') for s in a.subjects];root=resolve_project_root(a.platform,a.project_root)
    out=root/'derivatives'/'reports'/'group'/'EEG_channel_power_ratio_ROI';figs=out/'figures';figs.mkdir(parents=True,exist_ok=True)
    deriv=root/'data'/'BIDS'/'derivatives'/'group'/'EEG_channel_power_ratio_ROI';deriv.mkdir(parents=True,exist_ok=True)
    rid='channel_power_ratio_ROI_'+'_'.join(subjects);report=ParticipantPDF(str(out),rid)
    report.add_text('Analysis definition','Alternative analysis kept separate from the main all-channel group pipeline. For each participant, TFR power is calculated separately for each posterior channel in stim and no-stim. A normalized ratio (stim - no-stim)/(stim + no-stim) is then calculated separately for each channel from original unbaselined power. The participant ROI result is the arithmetic mean of these channel-wise ratios: ROI3 uses PO3/POz/PO4 and ROI8 uses PO3/POz/PO4/O1/Oz/O2/PO7/PO8. Finally, participant ROI ratios are averaged across subjects. Missing/rejected sensors are omitted and not interpolated.','Analysis')
    report.add_text('Subjects included',f'n={len(subjects)}\n'+', '.join('sub-'+s for s in subjects),'Analysis')
    roi3_values=[];roi8_values=[];used3={};used8={};available={}
    for s in subjects:
        print(f'Processing sub-{s}')
        ep=load(root,s,a);picks=common_roi_channels(ep);available[s]=picks
        if not picks:continue
        st=tfr(ep['stim'],picks,a.n_jobs);no=tfr(ep['no-stim'],picks,a.n_jobs);r=channel_ratio(st,no)
        r3,p3=participant_roi_ratio(r,ROI3);r8,p8=participant_roi_ratio(r,ROI8)
        used3[s]=p3;used8[s]=p8
        if r3 is not None:roi3_values.append(r3)
        if r8 is not None:roi8_values.append(r8)
    g3=grand_mean(roi3_values);g8=grand_mean(roi8_values)
    report.add_text('Channels contributing within each participant','ROI3:\n'+'\n'.join(f"sub-{s}: {', '.join(used3.get(s,[])) or 'none'}" for s in subjects)+'\n\nROI8:\n'+'\n'.join(f"sub-{s}: {', '.join(used8.get(s,[])) or 'none'}" for s in subjects),'Analysis')
    if g3 is not None:
        f,v=plot(g3,'Group mean of participant channel-wise ratios: PO3/POz/PO4');report.add_figure(f,str(figs/'group_ratio_mean_ROI3.png'),'Group ratio: mean PO3/POz/PO4 channel ratios',f'Each participant: channel power -> channel ratio -> mean available ROI3 channel ratios. Group: mean across n={len(roi3_values)} participant ROI ratios. Robust symmetric scale {v}.','Results')
    if g8 is not None:
        f,v=plot(g8,'Group mean of participant channel-wise ratios: 8 posterior channels');report.add_figure(f,str(figs/'group_ratio_mean_ROI8.png'),'Group ratio: mean 8-posterior-channel ratios',f'Each participant: channel power -> channel ratio -> mean available ROI8 channel ratios. Group: mean across n={len(roi8_values)} participant ROI ratios. Robust symmetric scale {v}.','Results')
    manuscript=f'Cue-locked time-frequency power was estimated separately for each retained posterior EEG sensor and stimulation condition using multitaper decomposition from 2 to 31.5 Hz in 0.5-Hz steps, n_cycles equal to frequency/2, time-bandwidth 2, FFT enabled, zero-mean tapers, trial averaging, and decimation by {DECIM}. For each participant and sensor, the normalized stimulation contrast was calculated from unbaselined power as (P_stim - P_no-stim)/(P_stim + P_no-stim). Sensor-wise normalized contrasts were then averaged within participant over PO3, POz and PO4 for the three-sensor ROI and over PO3, POz, PO4, O1, Oz, O2, PO7 and PO8 for the eight-sensor ROI, using only retained sensors available in both conditions. Finally, participant-level ROI contrasts were averaged across participants, giving equal weight to each participant with an available ROI.'
    report.add_text('Analysis report - manuscript style',manuscript,'Analysis')
    meta={'subjects':['sub-'+s for s in subjects],'ROI3':ROI3,'ROI8':ROI8,'available_posterior_by_subject':available,'ROI3_channels_by_subject':used3,'ROI8_channels_by_subject':used8,'order':'channel power -> channel ratio -> within-subject mean of channel ratios -> across-subject mean','ratio_baseline':'none','n_ROI3_subjects':len(roi3_values),'n_ROI8_subjects':len(roi8_values)}
    (deriv/f'{rid}_analysis.json').write_text(json.dumps(meta,indent=2)+'\n');print(f'Report: {report.pdf_fname}')
if __name__=='__main__':main()
