"""A02: TFR analysis in scalp-layout and posterior-channel views.

For each result, the script makes: (1) separate TFR traces for every common good
sensor arranged according to the scalp montage; (2) PO3, POz, PO4 separately.
Only the normalized ratio additionally gets the arithmetic mean of the posterior
channels. This keeps the posterior mean focused on the requested ratio.

Parameters intentionally match EEG_posterior_channels/sensor/A02_three_channel_TFR.py:
multitaper, 2-30 Hz in 1-Hz steps, n_cycles=freq/2, time_bandwidth=2, use_fft=True,
return_itc=False, average=True, decim=2; condition-specific TFRs use percent
baseline -0.3 to -0.1 s; comparative TFRs use unbaselined power and no baseline.
"""
from __future__ import annotations
import argparse,json
import matplotlib.pyplot as plt
import mne
import numpy as np
from pipeline_config import CONDITIONS,qc_dir,resolve_project_root,stage_path,subject_deriv_dir
from all_channel_report import participant_report,figure_dir,fmt_channels

POSTERIOR=('PO3','POz','PO4')
BASELINE=(-0.3,-0.1)
FREQS=np.arange(2.,31.,1.)
N_CYCLES=FREQS/2.
TIME_BANDWIDTH=2.
DECIM=2
PLOT_TMIN=-0.3
PLOT_TMAX=1.4
ROBUST_PERCENTILE=98.


def parse_args():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--subject',required=True); p.add_argument('--session',default='01'); p.add_argument('--task',default='SpAtt'); p.add_argument('--run',default='01'); p.add_argument('--platform',choices=['mac','bluebear'],default='mac'); p.add_argument('--project-root',default=None); p.add_argument('--n-jobs',type=int,default=4); return p.parse_args()
def load_epochs(root,s,a):
 d={}
 for c in CONDITIONS:
  p=stage_path(root,s,a.session,a.task,a.run,c,'clean','epo');
  if not p.exists(): raise FileNotFoundError(f'Missing final cleaned epochs for {c}: {p}')
  ep=mne.read_epochs(p,preload=True); keep=[k for k in ('cue_onset_right','cue_onset_left') if k in ep.event_id]; d[c]=ep[keep] if keep else ep
 return d
def common_good(d):
 s=d['stim']; n=d['no-stim']; return [ch for ch in s.copy().pick('eeg').ch_names if ch in n.ch_names and ch not in s.info['bads'] and ch not in n.info['bads']]
def compute(ep,jobs): return ep.compute_tfr(method='multitaper',freqs=FREQS,n_cycles=N_CYCLES,time_bandwidth=TIME_BANDWIDTH,use_fft=True,return_itc=False,average=True,decim=DECIM,n_jobs=jobs)
def robust_vlim(tfr):
 x=np.asarray(tfr.data); x=x[np.isfinite(x)];
 if x.size==0: return None
 vmax=float(np.percentile(np.abs(x),ROBUST_PERCENTILE));
 if not np.isfinite(vmax) or vmax<=0: return None
 return (-vmax,vmax)
def style_fig(fig):
 fig.patch.set_facecolor('white')
 for ax in fig.axes:
  ax.set_facecolor('white')
  ax.set_axisbelow(True)
 return fig
def scalp_topo(tfr,vlim):
 # plot_topo in the installed MNE version does not accept vlim. Create the
 # topographic layout first, then set image color limits on every TFR image.
 fig=style_fig(tfr.plot_topo(tmin=PLOT_TMIN,tmax=PLOT_TMAX,baseline=None,mode=None,cmap='RdBu_r',show=False))
 if vlim is not None:
  for ax in fig.axes:
   for im in ax.images: im.set_clim(*vlim)
 return fig
def posterior_separate(tfr,chs,prefix,vlim):
 fig,axes=plt.subplots(1,len(chs),figsize=(5*len(chs),4),constrained_layout=True); axes=[axes] if len(chs)==1 else list(axes)
 for ax,ch in zip(axes,chs):
  tfr.plot(picks=ch,tmin=PLOT_TMIN,tmax=PLOT_TMAX,baseline=None,mode=None,vlim=vlim,cmap='RdBu_r',axes=ax,show=False,colorbar=True); ax.set_facecolor('white'); ax.set_title(f'{prefix}: {ch}')
 return style_fig(fig)
def posterior_mean(tfr,chs,title,vlim):
 roi=tfr.copy().pick(chs); roi.data=roi.data.mean(axis=0,keepdims=True)
 info=mne.create_info(['Posterior_mean'],sfreq=roi.info['sfreq'],ch_types=['eeg']); roi.info=info
 fig=roi.plot(picks='Posterior_mean',tmin=PLOT_TMIN,tmax=PLOT_TMAX,baseline=None,mode=None,vlim=vlim,cmap='RdBu_r',show=False,colorbar=True); fig=fig[0] if isinstance(fig,list) else fig
 fig.axes[0].set_title(title); return style_fig(fig)
def add_views(report,figs,tfr,stem,title,caption,posterior,include_mean=False):
 vlim=robust_vlim(tfr); scale=f' Shared robust symmetric color scale: vmin={vlim[0]:.4g}, vmax={vlim[1]:.4g}.' if vlim else ' Automatic color scale used because robust limits could not be determined.'
 report.add_figure(scalp_topo(tfr,vlim),str(figs/f'{stem}_sensor_topography.png'),f'{title}: all sensors separately in scalp layout',caption+scale+' Each common good EEG sensor has its own TFR panel positioned by electrode location.','Time-frequency analysis')
 if posterior:
  report.add_figure(posterior_separate(tfr,posterior,title,vlim),str(figs/f'{stem}_posterior_separate.png'),f'{title}: PO3, POz and PO4 separately',caption+scale+f' Posterior order: {fmt_channels(posterior)}.','Time-frequency analysis')
  if include_mean:
   report.add_figure(posterior_mean(tfr,posterior,f'{title}: posterior mean',vlim),str(figs/f'{stem}_posterior_mean.png'),f'{title}: mean of PO3, POz and PO4',caption+scale+f' Arithmetic sensor mean across: {fmt_channels(posterior)}.','Time-frequency analysis')
 return vlim
def main():
 a=parse_args(); s=a.subject.removeprefix('sub-'); root=resolve_project_root(a.platform,a.project_root); report=participant_report(root,s); figs=figure_dir(root,s); deriv=subject_deriv_dir(root,s)
 epochs=load_epochs(root,s,a); common=common_good(epochs); posterior=[ch for ch in POSTERIOR if ch in common]
 if not common: raise RuntimeError('No common good EEG sensors available.')
 raw={}; display={}; scales={}
 for c in CONDITIONS:
  raw[c]=compute(epochs[c].copy().pick(common),a.n_jobs); raw[c].save(deriv/f'sub-{s}_ses-{a.session}_task-{a.task}_run-{a.run}_{c}_desc-allchannels_tfr.h5',overwrite=True)
  display[c]=raw[c].copy(); display[c].apply_baseline(BASELINE,mode='percent')
  cname='Stimulation' if c=='stim' else 'No stimulation'; title=f'{cname}: combined attention-left/right TFR'
  scales[c]=add_views(report,figs,display[c],f'A02_{c}_TFR',title,f'Multitaper 2-30 Hz; n_cycles=f/2; time-bandwidth={TIME_BANDWIDTH:g}; FFT=True; ITC=False; trial-average=True; decim={DECIM}; percent baseline {BASELINE}; display {PLOT_TMIN} to {PLOT_TMAX} s.',posterior,include_mean=False)
 diff=raw['stim'].copy(); diff.data=raw['stim'].data-raw['no-stim'].data; diff.save(deriv/f'sub-{s}_ses-{a.session}_task-{a.task}_run-{a.run}_stim-minus-no-stim_desc-allchannels_tfr.h5',overwrite=True)
 scales['difference']=add_views(report,figs,diff,'A02_stim_minus_no_stim_TFR','TFR: stimulation - no stimulation','Computed from unbaselined multitaper power; NO baseline correction. 2-30 Hz; n_cycles=f/2; time-bandwidth=2; decim=2; display -0.3 to 1.4 s.',posterior,include_mean=False)
 ratio=raw['stim'].copy(); ratio.data=(raw['stim'].data-raw['no-stim'].data)/(raw['stim'].data+raw['no-stim'].data+np.finfo(float).eps); ratio.save(deriv/f'sub-{s}_ses-{a.session}_task-{a.task}_run-{a.run}_stim-normalized-difference_desc-allchannels_tfr.h5',overwrite=True)
 scales['ratio']=add_views(report,figs,ratio,'A02_stim_normalized_difference_TFR','TFR: (stim - no-stim) / (stim + no-stim)','Computed from unbaselined multitaper power; NO baseline correction. 2-30 Hz; n_cycles=f/2; time-bandwidth=2; decim=2; display -0.3 to 1.4 s.',posterior,include_mean=True)
 details={'subject':f'sub-{s}','epoch_original_window_s':[-0.5,1.6],'method':'multitaper','frequencies_hz':FREQS.tolist(),'n_cycles':'frequency / 2','time_bandwidth':TIME_BANDWIDTH,'use_fft':True,'return_itc':False,'average':True,'decim':DECIM,'condition_baseline_s':list(BASELINE),'condition_baseline_mode':'percent','comparative_baseline':None,'posterior_order':list(POSTERIOR),'posterior_available':posterior,'views':'all sensors separately in scalp layout; PO3/POz/PO4 separately; posterior mean only for normalized ratio','robust_scale_percentile':ROBUST_PERCENTILE,'vlims':scales}
 (qc_dir(root,s)/'A02_tfr_analysis.json').write_text(json.dumps(details,indent=2)+'\n',encoding='utf-8')
 report.add_text('TFR analysis details',f'Input: final cleaned cue epochs (-0.5 to +1.6 s); attention-left and attention-right trials combined within each stimulation condition.\nMultitaper power: 2-30 Hz in 1-Hz steps; n_cycles=f/2; time-bandwidth={TIME_BANDWIDTH:g}; FFT=True; ITC=False; average=True; decimation={DECIM}.\nStimulation and no-stimulation: percent baseline {BASELINE}; titles specify combined attention-left/right TFR.\nComparative difference and normalized ratio: calculated from unbaselined power; NO baseline correction.\nEach result: every common good sensor separately in scalp layout; posterior PO3, POz, PO4 separately. Posterior mean is plotted only for the normalized ratio. A shared robust symmetric color scale uses the {ROBUST_PERCENTILE:g}th percentile of absolute displayed values, with the same vlim for the scalp-layout and enlarged posterior plots.','Time-frequency analysis')
 print(f'TFR complete for sub-{s}. Updated PDF: {report.pdf_fname}')
if __name__=='__main__': main()
