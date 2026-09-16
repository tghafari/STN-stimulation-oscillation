#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Complete cue-locked all-channel EEG grand-average report.

IMPORTANT ROI ORDER OF OPERATIONS
---------------------------------
For ROI3 (PO3/POz/PO4) and ROI8 (PO3/POz/PO4/O1/Oz/O2/PO7/PO8), sensors are
averaged in the EEG time domain WITHIN each participant and condition first. A TFR
is then computed from that participant's ROI epochs. Stim/no-stim, difference and
normalized ratio are formed at participant level. Finally those participant-level
ROI results are averaged across participants. Thus the normalized group ROI is:
mean_subjects[(Pstim_subject - Pnostim_subject)/(Pstim_subject + Pnostim_subject)].
There is no ratio-of-group-means and no mean-of-channel-wise-ratios.

All report content is written only to:
derivatives/reports/group/EEG_all_channels_complete_grand_average/
"""
from __future__ import annotations
import argparse,csv,json,sys
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
ERP_BASELINE=(-.1,0.);ERP_LP=30.;ERP_TMIN=-.1;ERP_TMAX=1.;BASELINE=(-.3,-.1);FREQS=np.arange(2.,32.,.5);N_CYCLES=FREQS/2.;TIME_BANDWIDTH=2.;DECIM=2;TFR_TMIN=-.5;TFR_TMAX=1.5;ROBUST=98.
def args():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--subjects',nargs='+',required=True);p.add_argument('--session',default='01');p.add_argument('--task',default='SpAtt');p.add_argument('--run',default='01');p.add_argument('--platform',choices=['mac','bluebear'],default='mac');p.add_argument('--project-root',default=None);p.add_argument('--n-jobs',type=int,default=4);return p.parse_args()
def diff_choice():
 while True:
  x=input('\nApply percent baseline (-0.3,-0.1 s) before participant-level stim - no-stim ROI TFR? (y/n): ').strip().lower()
  if x in {'y','yes'}:return True
  if x in {'n','no'}:return False
def load(root,s,a):
 out={}
 for c in CONDITIONS:
  f=stage_path(root,s,a.session,a.task,a.run,c,'clean','epo')
  if not f.exists():raise FileNotFoundError(f)
  e=mne.read_epochs(f,preload=True);k=[x for x in ('cue_onset_right','cue_onset_left') if x in e.event_id];out[c]=e[k] if k else e
 return out
def good(pair):
 st,no=pair['stim'],pair['no-stim'];return[ch for ch in st.copy().pick('eeg').ch_names if ch in no.ch_names and ch not in st.info['bads'] and ch not in no.info['bads']]
def evoked(ep,picks):
 x=ep.copy().pick(picks).average();x.filter(None,ERP_LP);x.apply_baseline(ERP_BASELINE);x.crop(ERP_TMIN,ERP_TMAX);return x
def tf(ep,picks,jobs):return ep.copy().pick(picks).compute_tfr(method='multitaper',freqs=FREQS,n_cycles=N_CYCLES,time_bandwidth=TIME_BANDWIDTH,use_fft=True,zero_mean=True,return_itc=False,average=True,decim=DECIM,n_jobs=jobs)
def info_template(epochs,chs):
 first=next(iter(epochs.values()))['stim'];info=first.copy().pick('eeg').info;present=[c for c in chs if c in info.ch_names]
 if len(present)==len(chs):return mne.pick_info(info,[info.ch_names.index(c) for c in chs],copy=True)
 z=mne.create_info(chs,first.info['sfreq'],'eeg');
 try:z.set_montage(first.get_montage(),on_missing='ignore')
 except Exception:pass
 return z
def channel_grand(objs,by,info,chs,c,is_tfr=False):
 rows=[np.mean([objs[s][c].copy().pick([ch]).data[0] for s in by[ch]],0) for ch in chs];first=objs[by[chs[0]][0]][c]
 return mne.time_frequency.AverageTFRArray(info.copy(),np.asarray(rows),first.times,first.freqs,nave=len(objs)) if is_tfr else mne.EvokedArray(np.asarray(rows),info.copy(),tmin=first.times[0],nave=len(objs))
def roi_epochs(ep,chs):
 x=ep.copy().pick(chs);data=x.get_data().mean(axis=1,keepdims=True);info=mne.create_info(['ROI_mean'],x.info['sfreq'],'eeg');return mne.EpochsArray(data,info,events=x.events.copy(),event_id=x.event_id.copy(),tmin=x.tmin,verbose=False)
def build_roi_tfrs(epochs,subjects,roi,jobs):
 """EEG sensor mean -> subject ROI TFR, separately for stim/no-stim."""
 out={};used={}
 for s in subjects:
  av=[ch for ch in roi if ch in good(epochs[s])]
  if not av:continue
  used[s]=av;out[s]={c:tf(roi_epochs(epochs[s][c],av),['ROI_mean'],jobs) for c in CONDITIONS}
 return out,used
def roi_result(rois,subjects,result,diff_bl):
 """Form result WITHIN subject, then arithmetic mean across subjects."""
 xs=[];used=[]
 for s in subjects:
  if s not in rois:continue
  st=rois[s]['stim'].copy();no=rois[s]['no-stim'].copy()
  if result=='stim':x=st.apply_baseline(BASELINE,mode='percent')
  elif result=='no-stim':x=no.apply_baseline(BASELINE,mode='percent')
  elif result=='difference':
   if diff_bl:st.apply_baseline(BASELINE,mode='percent');no.apply_baseline(BASELINE,mode='percent')
   x=st;x.data=st.data-no.data
  elif result=='ratio':x=st;x.data=(st.data-no.data)/(st.data+no.data+np.finfo(float).eps)
  else:raise ValueError(result)
  xs.append(x);used.append(s)
 if not xs:return None,[]
 g=xs[0].copy();g.data=np.mean([x.data for x in xs],axis=0);g.nave=len(xs);return g,used
def roi_evoked(epochs,subjects,roi,c):
 xs=[];used=[]
 for s in subjects:
  av=[ch for ch in roi if ch in good(epochs[s])]
  if not av:continue
  x=roi_epochs(epochs[s][c],av).average();x.filter(None,ERP_LP);x.apply_baseline(ERP_BASELINE);x.crop(ERP_TMIN,ERP_TMAX);xs.append(x);used.append(s)
 if not xs:return None,[]
 g=xs[0].copy();g.data=np.mean([x.data for x in xs],0);g.nave=len(xs);return g,used
def vl(x):
 z=np.asarray(x.data)[...,((x.times>=TFR_TMIN)&(x.times<=TFR_TMAX))];z=z[np.isfinite(z)]
 if not z.size:return(None,None)
 m=float(np.percentile(np.abs(z),ROBUST));return(-m,m) if m>0 else(None,None)
def scalp_tfr(x,v):
 f=x.plot_topo(tmin=TFR_TMIN,tmax=TFR_TMAX,fmin=2,fmax=31.5,baseline=None,mode=None,cmap='RdBu_r',show=False);f.set_size_inches(22,17)
 if None not in v:
  for ax in f.axes:
   for im in ax.images:im.set_clim(*v)
 return f
def eight_tfr(x,chs,counts,v):
 f,axs=plt.subplots(2,4,figsize=(20,9),constrained_layout=True)
 for ax,ch in zip(axs.ravel(),ROI8):
  if ch not in chs:ax.axis('off');continue
  kw=dict(picks=[ch],tmin=TFR_TMIN,tmax=TFR_TMAX,fmin=2,fmax=31.5,axes=ax,show=False,colorbar=True,cmap='RdBu_r');
  if None not in v:kw['vlim']=v
  x.plot(**kw);ax.set_title(f'{ch} (n={counts[ch]})')
 return f
def roi_plot(x,title):
 v=vl(x);kw=dict(picks=['ROI_mean'],tmin=TFR_TMIN,tmax=TFR_TMAX,fmin=2,fmax=31.5,show=False,colorbar=True,cmap='RdBu_r');
 if None not in v:kw['vlim']=v
 f=x.plot(**kw);f=f[0] if isinstance(f,list) else f;f.axes[0].set_title(title);return f
def eight_erp(compare,chs,counts):
 f,axs=plt.subplots(2,4,figsize=(20,9),constrained_layout=True)
 for ax,ch in zip(axs.ravel(),ROI8):
  if ch not in chs:ax.axis('off');continue
  mne.viz.plot_compare_evokeds(compare,picks=ch,axes=ax,show=False,ci=False);ax.set_xlim(ERP_TMIN,ERP_TMAX);ax.set_title(f'{ch} (n={counts[ch]})')
 return f
def roi_erp_plot(no,st,title):
 f=mne.viz.plot_compare_evokeds({'No stimulation':no,'Stimulation':st},picks=['ROI_mean'],show=False,ci=False);f=f[0] if isinstance(f,list) else f;f.axes[0].set_title(title);return f
def add_tfr(report,figs,x,roi3,roi8,subjects,by,chs,result,stem,title,details,diff_bl):
 v=vl(x);counts={ch:len(by[ch]) for ch in chs};section='TFR - '+title
 report.add_text('Analysis details',details+' Individual-channel scalp/8-sensor results are sensor-power analyses. ROI3 and ROI8 use a different explicitly defined order: EEG sensors are averaged within participant first, then participant ROI TFR is calculated, then the contrast/ratio is calculated within participant, and finally participant results are averaged.',section)
 report.add_figure(scalp_tfr(x,v),str(figs/f'{stem}_scalp.png'),title+': all-channel scalp layout',details,section);report.add_figure(eight_tfr(x,[c for c in ROI8 if c in chs],counts,v),str(figs/f'{stem}_8channels.png'),title+': eight posterior sensors',details,section)
 for rset,name,n in ((roi3,'PO3/POz/PO4',3),(roi8,'8 posterior sensors',8)):
  r,u=roi_result(rset,subjects,result,diff_bl)
  if r is not None:report.add_figure(roi_plot(r,f'{title}: mean {name}'),str(figs/f'{stem}_mean{n}.png'),f'{title}: mean {name}',details+f' Participant-level ROI result formed first, then averaged across n={len(u)} participants.',section)
def main():
 a=args();subjects=[s.removeprefix('sub-') for s in a.subjects];diff_bl=diff_choice();root=resolve_project_root(a.platform,a.project_root);out=root/'derivatives'/'reports'/'group'/'EEG_all_channels_complete_grand_average';figs=out/'figures';figs.mkdir(parents=True,exist_ok=True);deriv=root/'data'/'BIDS'/'derivatives'/'group'/'EEG_all_channels_complete_grand_average';deriv.mkdir(parents=True,exist_ok=True);rid='complete_grand_average_'+'_'.join(subjects);report=ParticipantPDF(str(out),rid)
 epochs={s:load(root,s,a) for s in subjects};goods={s:good(epochs[s]) for s in subjects};chs=[]
 for s in subjects:
  for ch in goods[s]:
   if ch not in chs:chs.append(ch)
 by={ch:[s for s in subjects if ch in goods[s]] for ch in chs};info=info_template(epochs,chs);chs=info.ch_names;by={ch:by[ch] for ch in chs}
 report.add_text('Subjects included',f'n={len(subjects)}\n'+', '.join('sub-'+s for s in subjects),'Group overview');report.add_text('Subjects contributing to each EEG channel','\n'.join(f"{ch} (n={len(by[ch])}): "+', '.join('sub-'+s for s in by[ch]) for ch in chs),'Group overview')
 with (deriv/f'{rid}_subjects_by_channel.csv').open('w',newline='') as f:w=csv.writer(f);w.writerow(['channel','n_subjects','subjects']);[w.writerow([ch,len(by[ch]),';'.join(by[ch])]) for ch in chs]
 # ERP individual sensors + EEG-domain ROI means
 sev={s:{c:evoked(epochs[s][c],goods[s]) for c in CONDITIONS} for s in subjects};gev={c:channel_grand(sev,by,info,chs,c) for c in CONDITIONS};compare={'No stimulation':gev['no-stim'],'Stimulation':gev['stim']};counts={ch:len(by[ch]) for ch in chs}
 report.add_text('ERP analysis details',f'Cue-locked, low-pass {ERP_LP:g} Hz, baseline {ERP_BASELINE}. Each sensor uses only subjects retaining that sensor in both conditions. ROI means average EEG sensors within participant before ERP averaging.','ERP');report.add_figure(eight_erp(compare,[c for c in ROI8 if c in chs],counts),str(figs/'ERP_8posterior.png'),'ERP: eight posterior sensors','2 x 4 posterior sensor view.','ERP')
 for roi,name,n in ((ROI3,'PO3/POz/PO4',3),(ROI8,'8 posterior sensors',8)):
  no,u=roi_evoked(epochs,subjects,roi,'no-stim');st,_=roi_evoked(epochs,subjects,roi,'stim')
  if no is not None:report.add_figure(roi_erp_plot(no,st,'ERP: mean '+name),str(figs/f'ERP_mean{n}.png'),'ERP: mean '+name,f'EEG sensors averaged within participant first; n={len(u)}.','ERP')
 # TFR individual sensors
 stfr={s:{c:tf(epochs[s][c],goods[s],a.n_jobs) for c in CONDITIONS} for s in subjects};raw={c:channel_grand(stfr,by,info,chs,c,True) for c in CONDITIONS};no=raw['no-stim'].copy().apply_baseline(BASELINE,mode='percent');st=raw['stim'].copy().apply_baseline(BASELINE,mode='percent');sd=raw['stim'].copy();nd=raw['no-stim'].copy()
 if diff_bl:sd.apply_baseline(BASELINE,mode='percent');nd.apply_baseline(BASELINE,mode='percent')
 diff=sd.copy();diff.data=sd.data-nd.data;ratio=raw['stim'].copy();ratio.data=(raw['stim'].data-raw['no-stim'].data)/(raw['stim'].data+raw['no-stim'].data+np.finfo(float).eps)
 # ROI TFRs: EEG mean BEFORE TFR; computed once and reused.
 roi3,roi3_used=build_roi_tfrs(epochs,subjects,ROI3,a.n_jobs);roi8,roi8_used=build_roi_tfrs(epochs,subjects,ROI8,a.n_jobs)
 report.add_text('Posterior ROI channel availability','ROI3 (PO3/POz/PO4):\n'+'\n'.join(f"sub-{s}: {', '.join(roi3_used.get(s,[])) or 'none'}" for s in subjects)+'\n\nROI8:\n'+'\n'.join(f"sub-{s}: {', '.join(roi8_used.get(s,[])) or 'none'}" for s in subjects),'Group overview')
 common=f'Multitaper 2-31.5 Hz, 0.5-Hz steps; n_cycles=f/2; time-bandwidth={TIME_BANDWIDTH:g}; FFT=True; zero_mean=True; ITC=False; trial-average=True; decim={DECIM}; cue onset=0 s.'
 add_tfr(report,figs,no,roi3,roi8,subjects,by,chs,'no-stim','TFR_no_stim','No stimulation TFR',common+f' Percent baseline {BASELINE}.',diff_bl);add_tfr(report,figs,st,roi3,roi8,subjects,by,chs,'stim','TFR_stim','Stimulation TFR',common+f' Percent baseline {BASELINE}.',diff_bl);add_tfr(report,figs,diff,roi3,roi8,subjects,by,chs,'difference','TFR_difference','Stimulation - no stimulation TFR',common+f' Participant-level difference baseline correction={diff_bl}.',diff_bl);add_tfr(report,figs,ratio,roi3,roi8,subjects,by,chs,'ratio','TFR_ratio','(Stimulation - no stimulation)/(stimulation + no stimulation)',common+' ROI ratio is computed from unbaselined participant ROI power.',diff_bl)
 manuscript=f'Final cleaned cue-locked EEG epochs from {len(subjects)} participants were analysed. Cue-left and cue-right trials were combined within stimulation condition. For sensor-wise analyses, participant trial averages/TFRs were calculated first and each sensor included only participants retaining that sensor in both conditions. For posterior ROI analyses, available ROI EEG sensors were averaged in the time domain within each participant and condition before TFR estimation. Thus ROI power was estimated from the mean EEG signal rather than by averaging sensor powers. For each participant, the normalized ROI contrast was then calculated as (stimulation power - no-stimulation power)/(stimulation power + no-stimulation power) from unbaselined participant-level ROI TFRs. The reported group normalized ROI contrast is the arithmetic mean of these participant-level ratios; it is not a ratio of group means. The same order was used for the 3-sensor ROI (PO3, POz, PO4) and 8-sensor ROI (PO3, POz, PO4, O1, Oz, O2, PO7, PO8). TFRs used multitaper decomposition from 2 to 31.5 Hz in 0.5-Hz steps, n_cycles=f/2, time-bandwidth={TIME_BANDWIDTH:g}, FFT=True, zero_mean=True, ITC=False, trial averaging and decimation={DECIM}. Condition TFRs were percent-baseline corrected using {BASELINE}. Difference baseline correction was set to {diff_bl}.';report.add_text('Analysis report - manuscript style',manuscript,'Analysis');report.add_text('Exact analysis parameters',f'Subjects: '+', '.join('sub-'+s for s in subjects)+f'\nROI3={ROI3}\nROI8={ROI8}\nROI order: mean EEG -> subject TFR -> subject contrast/ratio -> mean subjects.\nRatio baseline: none. Difference baseline={diff_bl}.','Analysis')
 (deriv/f'{rid}_analysis.json').write_text(json.dumps({'subjects':subjects,'subjects_by_channel':by,'ROI3_channels_by_subject':roi3_used,'ROI8_channels_by_subject':roi8_used,'ROI_order':'mean EEG -> subject TFR -> subject contrast/ratio -> mean subjects','difference_baseline':diff_bl},indent=2)+'\n');print(f'Complete report only: {report.pdf_fname}')
if __name__=='__main__':main()
