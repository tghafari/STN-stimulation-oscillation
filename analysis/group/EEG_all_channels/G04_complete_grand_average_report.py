#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""G04: complete cue-locked all-channel EEG grand-average report.

Each participant is averaged first. For each individual sensor, only participants
retaining that sensor in BOTH conditions contribute. Missing sensors are not
interpolated at group level.

ERP report views:
  1. large readable all-channel scalp layout
  2. eight posterior sensors (PO3, POz, PO4, O1, Oz, O2, PO7, PO8) in TWO rows
     with four sensors per row (2 x 4)
  3. mean of the eight posterior sensors
  4. mean of PO3/POz/PO4

Every TFR section (no-stim, stim, stim-no-stim, normalized difference) contains:
  1. all-channel scalp layout
  2. eight posterior sensors in 2 x 4 form
  3. mean of PO3/POz/PO4
  4. mean of all eight posterior sensors

ROI means are computed within participant first, then participant ROI averages are
grand-averaged. TFR parameters match the posterior grand-average analysis.
"""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import matplotlib.pyplot as plt
import mne
import numpy as np
HERE=Path(__file__).resolve().parent; ANALYSIS_DIR=HERE.parents[1]
for p in (ANALYSIS_DIR/'subject'/'EEG_all_channels',ANALYSIS_DIR/'utils'):
 if str(p) not in sys.path:sys.path.insert(0,str(p))
from pipeline_config import CONDITIONS,resolve_project_root,stage_path
from pdf_report import ParticipantPDF
ROI8=('PO3','POz','PO4','O1','Oz','O2','PO7','PO8'); ROI3=('PO3','POz','PO4')
ERP_BASELINE=(-.1,0.);ERP_LP=30.;ERP_TMIN=-.1;ERP_TMAX=1.
BASELINE=(-.3,-.1);FREQS=np.arange(2.,32.,.5);N_CYCLES=FREQS/2.;TIME_BANDWIDTH=2.;DECIM=2;TFR_TMIN=-.5;TFR_TMAX=1.5;ROBUST=98.
def args():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--subjects',nargs='+',required=True);p.add_argument('--session',default='01');p.add_argument('--task',default='SpAtt');p.add_argument('--run',default='01');p.add_argument('--platform',choices=['mac','bluebear'],default='mac');p.add_argument('--project-root',default=None);p.add_argument('--n-jobs',type=int,default=4);return p.parse_args()
def diff_baseline_choice():
 while True:
  x=input('\nApply percent baseline (-0.3,-0.1 s) before calculating stim - no-stim TFR? (y/n): ').strip().lower()
  if x in {'y','yes'}:return True
  if x in {'n','no'}:return False
def load_epochs(root,s,a):
 out={}
 for c in CONDITIONS:
  p=stage_path(root,s,a.session,a.task,a.run,c,'clean','epo')
  if not p.exists():raise FileNotFoundError(p)
  ep=mne.read_epochs(p,preload=True);keep=[k for k in ('cue_onset_right','cue_onset_left') if k in ep.event_id];out[c]=ep[keep] if keep else ep
 return out
def good_channels(pair):
 s=pair['stim'];n=pair['no-stim'];return[ch for ch in s.copy().pick('eeg').ch_names if ch in n.ch_names and ch not in s.info['bads'] and ch not in n.info['bads']]
def evoked(ep,picks):
 x=ep.copy().pick(picks).average();x.filter(None,ERP_LP);x.apply_baseline(ERP_BASELINE);x.crop(ERP_TMIN,ERP_TMAX);return x
def tfr(ep,picks,jobs):return ep.copy().pick(picks).compute_tfr(method='multitaper',freqs=FREQS,n_cycles=N_CYCLES,time_bandwidth=TIME_BANDWIDTH,use_fft=True,zero_mean=True,return_itc=False,average=True,decim=DECIM,n_jobs=jobs)
def template_info(epochs,channels):
 for pair in epochs.values():
  info=pair['stim'].copy().pick('eeg').info
  if all(ch in info.ch_names for ch in channels):return mne.pick_info(info,[info.ch_names.index(ch) for ch in channels],copy=True)
 first=next(iter(epochs.values()))['stim'];info=mne.create_info(channels,first.info['sfreq'],'eeg')
 try:info.set_montage(first.get_montage(),on_missing='ignore')
 except Exception:pass
 return info
def channel_grand(objs,by_ch,info,channels,c,is_tfr=False):
 rows=[np.mean([objs[s][c].copy().pick([ch]).data[0] for s in by_ch[ch]],axis=0) for ch in channels];first=objs[by_ch[channels[0]][0]][c]
 return mne.time_frequency.AverageTFRArray(info.copy(),np.asarray(rows),first.times,first.freqs,nave=len(objs),comment=f'grand {c}') if is_tfr else mne.EvokedArray(np.asarray(rows),info.copy(),tmin=float(first.times[0]),nave=len(objs),comment=f'grand {c}')
def roi_evoked(objs,subjects,roi,c):
 xs=[];used=[]
 for s in subjects:
  av=[ch for ch in roi if ch in objs[s]['stim'].ch_names and ch in objs[s]['no-stim'].ch_names]
  if not av:continue
  x=objs[s][c].copy().pick(av);xs.append(mne.EvokedArray(x.data.mean(0,keepdims=True),mne.create_info(['ROI_mean'],x.info['sfreq'],'eeg'),tmin=x.times[0]));used.append(s)
 if not xs:return None,[]
 g=xs[0].copy();g.data=np.mean([x.data for x in xs],axis=0);g.nave=len(xs);return g,used
def roi_tfr(objs,subjects,roi,result,diff_bl):
 xs=[];used=[]
 for s in subjects:
  av=[ch for ch in roi if ch in objs[s]['stim'].ch_names and ch in objs[s]['no-stim'].ch_names]
  if not av:continue
  st=objs[s]['stim'].copy().pick(av);no=objs[s]['no-stim'].copy().pick(av)
  if result=='stim':x=st.apply_baseline(BASELINE,mode='percent')
  elif result=='no-stim':x=no.apply_baseline(BASELINE,mode='percent')
  elif result=='difference':
   if diff_bl:st.apply_baseline(BASELINE,mode='percent');no.apply_baseline(BASELINE,mode='percent')
   x=st;x.data=st.data-no.data
  else:x=st;x.data=(st.data-no.data)/(st.data+no.data+np.finfo(float).eps)
  x.data=x.data.mean(0,keepdims=True);x.info=mne.create_info(['ROI_mean'],x.info['sfreq'],'eeg');xs.append(x);used.append(s)
 if not xs:return None,[]
 g=xs[0].copy();g.data=np.mean([x.data for x in xs],axis=0);g.nave=len(xs);return g,used
def vlim(x):
 ti=(x.times>=TFR_TMIN)&(x.times<=TFR_TMAX);z=np.asarray(x.data)[...,ti];z=z[np.isfinite(z)]
 if not z.size:return(None,None)
 m=float(np.percentile(np.abs(z),ROBUST));return(-m,m) if np.isfinite(m) and m>0 else(None,None)
def style(fig):
 fig.patch.set_facecolor('white')
 for ax in fig.axes:ax.set_facecolor('white')
 return fig
def scalp_erp(compare,channels):
 # Very large canvas: report PDF will scale it down as one image while retaining
 # substantially more pixels/detail per electrode than the previous 18x14 figure.
 fig=mne.viz.plot_compare_evokeds(compare,picks=channels,combine=None,axes='topo',show=False,ci=False,truncate_xaxis=False,truncate_yaxis=False,legend=True);fig=fig[0] if isinstance(fig,list) else fig;fig.set_size_inches(28,22,forward=True)
 # Keep more of each topo panel than before (.90 rather than .72), because the much
 # larger canvas provides separation without making individual waveforms tiny.
 for ax in fig.axes:
  box=ax.get_position();cx=box.x0+box.width/2;cy=box.y0+box.height/2;w=box.width*.90;h=box.height*.90;ax.set_position([cx-w/2,cy-h/2,w,h]);ax.tick_params(labelsize=8)
 return style(fig)
def scalp_tfr(x,lim):
 fig=x.plot_topo(tmin=TFR_TMIN,tmax=TFR_TMAX,fmin=2,fmax=31.5,baseline=None,mode=None,cmap='RdBu_r',show=False);fig.set_size_inches(22,17,forward=True)
 for ax in fig.axes:
  if None not in lim:
   for im in ax.images:im.set_clim(*lim)
 return style(fig)
def eight_erp(compare,available,counts):
 # User requested two rows for the eight sensors: 4 sensors in each row.
 fig,axs=plt.subplots(2,4,figsize=(20,9),constrained_layout=True)
 for ax,ch in zip(axs.ravel(),ROI8):
  if ch not in available:ax.axis('off');ax.set_title(f'{ch}: unavailable');continue
  mne.viz.plot_compare_evokeds(compare,picks=ch,axes=ax,show=False,ci=False,truncate_xaxis=False,truncate_yaxis=False);ax.axvline(0,color='k',ls='--',lw=.8);ax.set_xlim(ERP_TMIN,ERP_TMAX);ax.set_title(f'{ch} (n={counts[ch]})')
 fig.suptitle('Grand-average ERP: eight posterior sensors (2 rows x 4 sensors)',fontsize=15);return fig
def eight_tfr(x,available,counts,lim):
 fig,axs=plt.subplots(2,4,figsize=(20,9),constrained_layout=True)
 for ax,ch in zip(axs.ravel(),ROI8):
  if ch not in available:ax.axis('off');ax.set_title(f'{ch}: unavailable');continue
  kw=dict(picks=[ch],tmin=TFR_TMIN,tmax=TFR_TMAX,fmin=2,fmax=31.5,baseline=None,mode=None,axes=ax,show=False,colorbar=True,cmap='RdBu_r')
  if None not in lim:kw['vlim']=lim
  x.plot(**kw);ax.axvline(0,color='k',ls='--',lw=.8);ax.set_title(f'{ch} (n={counts[ch]})')
 return fig
def roi_erp_plot(no,st,title):
 fig=mne.viz.plot_compare_evokeds({'No stimulation':no,'Stimulation':st},picks=['ROI_mean'],show=False,ci=False,truncate_xaxis=False,truncate_yaxis=False);fig=fig[0] if isinstance(fig,list) else fig;fig.axes[0].axvline(0,color='k',ls='--');fig.axes[0].set_title(title);return fig
def roi_tfr_plot(x,title,lim):
 kw=dict(picks=['ROI_mean'],tmin=TFR_TMIN,tmax=TFR_TMAX,fmin=2,fmax=31.5,baseline=None,mode=None,show=False,colorbar=True,cmap='RdBu_r')
 if None not in lim:kw['vlim']=lim
 fig=x.plot(**kw);fig=fig[0] if isinstance(fig,list) else fig;fig.axes[0].set_title(title);return fig
def add_erp(report,figs,grand,objs,subjects,by_ch,channels):
 compare={'No stimulation':grand['no-stim'],'Stimulation':grand['stim']};counts={ch:len(by_ch[ch]) for ch in channels};av8=[ch for ch in ROI8 if ch in channels]
 report.add_text('ERP analysis details',f'Cue-locked ERP; cue-left/right combined; subject trial averages formed first; low-pass={ERP_LP:g} Hz; baseline={ERP_BASELINE}; display={ERP_TMIN} to {ERP_TMAX} s. Sensor-specific grand averages contain only subjects retaining that sensor in both conditions.','ERP')
 report.add_figure(scalp_erp(compare,channels),str(figs/'ERP_all_channels_scalp_LARGE.png'),'ERP: all channels in large scalp layout','Large 28 x 22 inch source figure so individual sensor waveforms remain readable in the report.','ERP')
 report.add_figure(eight_erp(compare,av8,counts),str(figs/'ERP_eight_posterior_2rows.png'),'ERP: eight posterior sensors in two rows','Two rows x four sensors: PO3, POz, PO4, O1 / Oz, O2, PO7, PO8.','ERP')
 for roi,name in ((ROI8,'8 posterior sensors'),(ROI3,'PO3/POz/PO4')):
  no,u=roi_evoked(objs,subjects,roi,'no-stim');st,_=roi_evoked(objs,subjects,roi,'stim')
  if no is not None:report.add_figure(roi_erp_plot(no,st,f'ERP: mean {name}'),str(figs/f'ERP_mean_{len(roi)}_posterior.png'),f'ERP: mean {name}',f'Channels averaged within subject first, then subjects grand-averaged (n={len(u)}).','ERP')
def add_tfr_result(report,figs,x,objs,subjects,by_ch,channels,result,stem,title,details,diff_bl):
 lim=vlim(x);counts={ch:len(by_ch[ch]) for ch in channels};av8=[ch for ch in ROI8 if ch in channels];section=f'TFR - {title}';scale=f'Robust symmetric scale {lim[0]:.4g} to {lim[1]:.4g}.' if None not in lim else 'Automatic scale.'
 report.add_text('Analysis details',details+' '+scale+' Output order: scalp layout; eight posterior sensors; mean PO3/POz/PO4; mean eight posterior sensors.',section)
 report.add_figure(scalp_tfr(x,lim),str(figs/f'{stem}_all_channels_scalp.png'),f'{title}: all-channel scalp layout',details+' '+scale,section)
 report.add_figure(eight_tfr(x,av8,counts,lim),str(figs/f'{stem}_eight_posterior_2x4.png'),f'{title}: eight posterior sensors',details+' '+scale,section)
 # Explicit order requested: 3-channel mean first, 8-channel mean second.
 for roi,name in ((ROI3,'PO3/POz/PO4'),(ROI8,'8 posterior sensors')):
  r,u=roi_tfr(objs,subjects,roi,result,diff_bl)
  if r is not None:
   rl=vlim(r);report.add_figure(roi_tfr_plot(r,f'{title}: mean {name}',rl),str(figs/f'{stem}_mean_{len(roi)}_posterior.png'),f'{title}: mean {name}',details+f' Channels averaged within participant first, then participant ROI TFRs grand-averaged (n={len(u)}).',section)
def main():
 a=args();subjects=[s.removeprefix('sub-') for s in a.subjects];diff_bl=diff_baseline_choice();root=resolve_project_root(a.platform,a.project_root);out=root/'derivatives'/'reports'/'group'/'EEG_all_channels_complete_grand_average';figs=out/'figures';figs.mkdir(parents=True,exist_ok=True);deriv=root/'data'/'BIDS'/'derivatives'/'group'/'EEG_all_channels_complete_grand_average';deriv.mkdir(parents=True,exist_ok=True);rid='complete_grand_average_'+'_'.join(subjects);report=ParticipantPDF(str(out),rid)
 epochs={s:load_epochs(root,s,a) for s in subjects};good={s:good_channels(epochs[s]) for s in subjects};channels=[]
 for s in subjects:
  for ch in good[s]:
   if ch not in channels:channels.append(ch)
 by_ch={ch:[s for s in subjects if ch in good[s]] for ch in channels};info=template_info(epochs,channels);channels=info.ch_names;by_ch={ch:by_ch[ch] for ch in channels}
 report.add_text('Subjects included',f'n={len(subjects)}\n'+', '.join('sub-'+s for s in subjects),'Group overview');report.add_text('Subjects contributing to each EEG channel','\n'.join(f"{ch} (n={len(by_ch[ch])}): "+', '.join('sub-'+s for s in by_ch[ch]) for ch in channels),'Group overview')
 with (deriv/f'{rid}_subjects_by_channel.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.writer(f);w.writerow(['channel','n_subjects','subjects']);[w.writerow([ch,len(by_ch[ch]),';'.join('sub-'+s for s in by_ch[ch])]) for ch in channels]
 sev={s:{c:evoked(epochs[s][c],good[s]) for c in CONDITIONS} for s in subjects};gev={c:channel_grand(sev,by_ch,info,channels,c) for c in CONDITIONS};add_erp(report,figs,gev,sev,subjects,by_ch,channels)
 objs={}
 for s in subjects:
  objs[s]={}
  for c in CONDITIONS:print(f'Computing TFR sub-{s} {c}');objs[s][c]=tfr(epochs[s][c],good[s],a.n_jobs)
 raw={c:channel_grand(objs,by_ch,info,channels,c,True) for c in CONDITIONS};no=raw['no-stim'].copy().apply_baseline(BASELINE,mode='percent');st=raw['stim'].copy().apply_baseline(BASELINE,mode='percent');sd=raw['stim'].copy();nd=raw['no-stim'].copy()
 if diff_bl:sd.apply_baseline(BASELINE,mode='percent');nd.apply_baseline(BASELINE,mode='percent')
 diff=sd.copy();diff.data=sd.data-nd.data;ratio=raw['stim'].copy();ratio.data=(raw['stim'].data-raw['no-stim'].data)/(raw['stim'].data+raw['no-stim'].data+np.finfo(float).eps);common=f'Multitaper 2-31.5 Hz, 0.5-Hz steps; n_cycles=f/2; time-bandwidth={TIME_BANDWIDTH:g}; FFT=True; zero_mean=True; ITC=False; average=True; decim={DECIM}; cue onset=0 s.'
 add_tfr_result(report,figs,no,objs,subjects,by_ch,channels,'no-stim','TFR_no_stim','No stimulation TFR',common+f' Percent baseline {BASELINE}.',diff_bl);add_tfr_result(report,figs,st,objs,subjects,by_ch,channels,'stim','TFR_stim','Stimulation TFR',common+f' Percent baseline {BASELINE}.',diff_bl);add_tfr_result(report,figs,diff,objs,subjects,by_ch,channels,'difference','TFR_difference','Stimulation - no stimulation TFR',common+f' Difference baseline correction={diff_bl}.',diff_bl);add_tfr_result(report,figs,ratio,objs,subjects,by_ch,channels,'ratio','TFR_ratio','(Stimulation - no stimulation) / (stimulation + no stimulation)',common+' Original unbaselined power; no baseline correction.',diff_bl)
 manuscript=f'Final cleaned cue-locked EEG epochs from {len(subjects)} participants were analysed. Cue-left and cue-right trials were combined within stimulation condition. No epochs were concatenated across participants. Each participant was averaged before group averaging. For each sensor, only participants retaining that sensor in both conditions contributed. ERPs were low-pass filtered at {ERP_LP:g} Hz and baseline corrected from {ERP_BASELINE[0]:g} to {ERP_BASELINE[1]:g} s. Time-frequency power was estimated from 2 to 31.5 Hz in 0.5-Hz steps using multitaper decomposition, n_cycles=f/2, time-bandwidth={TIME_BANDWIDTH:g}, FFT, zero-mean tapers, trial averaging and decimation={DECIM}. Condition TFRs used percent baseline {BASELINE}. The normalized contrast used original unbaselined power. Posterior summaries were calculated separately for PO3/POz/PO4 and for PO3/POz/PO4/O1/Oz/O2/PO7/PO8 by averaging available ROI sensors within participant before grand averaging participants.';report.add_text('Analysis report - manuscript style',manuscript,'Manuscript-style analysis report');(deriv/f'{rid}_analysis.json').write_text(json.dumps({'subjects':['sub-'+s for s in subjects],'subjects_by_channel':{ch:['sub-'+s for s in by_ch[ch]] for ch in channels},'ROI3':ROI3,'ROI8':ROI8,'difference_baseline':diff_bl},indent=2)+'\n');print(report.pdf_fname)
if __name__=='__main__':main()
