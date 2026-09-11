"""A02: all-channel and posterior TFR analysis with matched plotting scales.

Multitaper power is computed once per stimulation condition using the final cleaned
cue epochs: 2-30 Hz in 1-Hz steps, n_cycles=f/2, time_bandwidth=2, FFT=True,
ITC=False, trial average=True, decim=2. Attention-left/right cue trials are combined.

Stim and no-stim displays use percent baseline correction (-0.3,-0.1 s).
Comparisons are computed from unbaselined power and receive NO baseline correction.
All-channel scalp-layout and enlarged posterior plots use the SAME TFR object,
time/frequency limits, baseline state, colormap, and robust vlim for each result.
Posterior display order is PO3 | POz | PO4. A posterior mean is shown only for the
normalized ratio (stim-no-stim)/(stim+no-stim).
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
ROBUST_PERCENTILE=98.0


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subject',required=True); p.add_argument('--session',default='01')
    p.add_argument('--task',default='SpAtt'); p.add_argument('--run',default='01')
    p.add_argument('--platform',choices=['mac','bluebear'],default='mac'); p.add_argument('--project-root',default=None)
    p.add_argument('--n-jobs',type=int,default=4); return p.parse_args()


def load_epochs(root,s,a):
    d={}
    for c in CONDITIONS:
        p=stage_path(root,s,a.session,a.task,a.run,c,'clean','epo')
        if not p.exists(): raise FileNotFoundError(f'Missing final cleaned epochs: {p}')
        ep=mne.read_epochs(p,preload=True)
        keep=[k for k in ('cue_onset_right','cue_onset_left') if k in ep.event_id]
        d[c]=ep[keep] if keep else ep
    return d


def common_good(d):
    s=d['stim']; n=d['no-stim']
    return [ch for ch in s.copy().pick('eeg').ch_names if ch in n.ch_names and ch not in s.info['bads'] and ch not in n.info['bads']]


def compute(ep,jobs):
    return ep.compute_tfr(method='multitaper',freqs=FREQS,n_cycles=N_CYCLES,time_bandwidth=TIME_BANDWIDTH,use_fft=True,return_itc=False,average=True,decim=DECIM,n_jobs=jobs)


def robust_vlim(tfr,symmetric=True):
    """Robust common scale from the data actually displayed, avoiding outlier flattening."""
    ti=(tfr.times>=PLOT_TMIN)&(tfr.times<=PLOT_TMAX)
    x=np.asarray(tfr.data)[...,ti]
    finite=x[np.isfinite(x)]
    if finite.size==0: return (None,None)
    if symmetric:
        vmax=float(np.percentile(np.abs(finite),ROBUST_PERCENTILE))
        if not np.isfinite(vmax) or vmax==0: return (None,None)
        return (-vmax,vmax)
    lo=float(np.percentile(finite,100.-ROBUST_PERCENTILE)); hi=float(np.percentile(finite,ROBUST_PERCENTILE))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo==hi: return (None,None)
    return (lo,hi)


def scalp_topo(tfr,vlim):
    fig=tfr.plot_topo(tmin=PLOT_TMIN,tmax=PLOT_TMAX,baseline=None,mode=None,vlim=vlim,cmap='RdBu_r',show=False)
    fig.patch.set_facecolor('white')
    for ax in fig.axes: ax.set_facecolor('white')
    return fig


def posterior_separate(tfr,chs,prefix,vlim):
    fig,axes=plt.subplots(1,len(chs),figsize=(5*len(chs),4),constrained_layout=True)
    axes=[axes] if len(chs)==1 else list(axes)
    for ax,ch in zip(axes,chs):
        tfr.plot(picks=ch,tmin=PLOT_TMIN,tmax=PLOT_TMAX,baseline=None,mode=None,axes=ax,show=False,colorbar=True,vlim=vlim,cmap='RdBu_r')
        ax.set_title(f'{prefix}: {ch}')
    return fig


def posterior_mean(tfr,chs,title,vlim):
    roi=tfr.copy().pick(chs)
    roi.data=roi.data.mean(axis=0,keepdims=True)
    roi.info=mne.pick_info(roi.info,[0],copy=True)
    roi.info['chs'][0]['ch_name']='Posterior mean'; roi.info['ch_names'][0]='Posterior mean'
    fig=roi.plot(picks='Posterior mean',tmin=PLOT_TMIN,tmax=PLOT_TMAX,baseline=None,mode=None,show=False,colorbar=True,vlim=vlim,cmap='RdBu_r')
    fig=fig[0] if isinstance(fig,list) else fig
    fig.axes[0].set_title(title)
    return fig


def add_views(report,figs,tfr,stem,title,caption,posterior,include_mean=False):
    # One vlim is calculated once and reused for scalp-layout and posterior panels.
    # This makes a given channel visually identical whether seen in the all-channel
    # layout or in the enlarged posterior view.
    vlim=robust_vlim(tfr,symmetric=True)
    scale=f' Shared robust color scale: vmin={vlim[0]:.4g}, vmax={vlim[1]:.4g}.' if None not in vlim else ' Automatic color scale used because robust limits could not be determined.'
    report.add_figure(scalp_topo(tfr,vlim),str(figs/f'{stem}_sensor_topography.png'),f'{title}: all sensors separately in scalp layout',caption+scale+' Each common good EEG sensor has its own panel at its electrode location.','Time-frequency analysis')
    if posterior:
        report.add_figure(posterior_separate(tfr,posterior,title,vlim),str(figs/f'{stem}_posterior_separate.png'),f'{title}: PO3, POz and PO4 separately',caption+scale+f' Posterior order: {fmt_channels(posterior)}.','Time-frequency analysis')
        if include_mean:
            report.add_figure(posterior_mean(tfr,posterior,f'{title}: posterior mean',vlim),str(figs/f'{stem}_posterior_mean.png'),f'{title}: mean of PO3, POz and PO4',caption+scale+f' Arithmetic sensor mean across: {fmt_channels(posterior)}.','Time-frequency analysis')
    return vlim


def main():
    a=parse_args(); s=a.subject.removeprefix('sub-'); root=resolve_project_root(a.platform,a.project_root)
    report=participant_report(root,s); figs=figure_dir(root,s); deriv=subject_deriv_dir(root,s)
    epochs=load_epochs(root,s,a); common=common_good(epochs); posterior=[ch for ch in POSTERIOR if ch in common]
    if not common: raise RuntimeError('No common good EEG sensors available.')

    raw={}; display={}; scales={}
    for c in CONDITIONS:
        # Same TFR calculation for every sensor, including PO3/POz/PO4.
        raw[c]=compute(epochs[c].copy().pick(common),a.n_jobs)
        raw[c].save(deriv/f'sub-{s}_ses-{a.session}_task-{a.task}_run-{a.run}_{c}_desc-allchannels_tfr.h5',overwrite=True)
        display[c]=raw[c].copy().apply_baseline(BASELINE,mode='percent')
        condition_name='No stimulation' if c=='no-stim' else 'Stimulation'
        title=f'{condition_name}: combined attention-left/right TFR'
        caption=f'Multitaper 2-30 Hz; n_cycles=f/2; time-bandwidth={TIME_BANDWIDTH:g}; FFT=True; ITC=False; trial-average=True; decim={DECIM}; percent baseline {BASELINE}; display {PLOT_TMIN} to {PLOT_TMAX} s.'
        scales[c]=add_views(report,figs,display[c],f'A02_{c}_TFR',title,caption,posterior,include_mean=False)

    # Difference: same unbaselined raw power for all sensors; no posterior mean.
    diff=raw['stim'].copy(); diff.data=raw['stim'].data-raw['no-stim'].data
    diff.save(deriv/f'sub-{s}_ses-{a.session}_task-{a.task}_run-{a.run}_stim-minus-no-stim_desc-allchannels_tfr.h5',overwrite=True)
    scales['difference']=add_views(report,figs,diff,'A02_stim_minus_no_stim_TFR','TFR: stimulation - no stimulation',f'Computed from unbaselined multitaper power. NO baseline correction. 2-30 Hz; n_cycles=f/2; time-bandwidth={TIME_BANDWIDTH:g}; FFT=True; ITC=False; trial-average=True; decim={DECIM}; display {PLOT_TMIN} to {PLOT_TMAX} s.',posterior,include_mean=False)

    # Normalized ratio: posterior mean is requested here only.
    ratio=raw['stim'].copy(); denom=raw['stim'].data+raw['no-stim'].data
    ratio.data=(raw['stim'].data-raw['no-stim'].data)/(denom+np.finfo(float).eps)
    ratio.save(deriv/f'sub-{s}_ses-{a.session}_task-{a.task}_run-{a.run}_stim-normalized-difference_desc-allchannels_tfr.h5',overwrite=True)
    scales['ratio']=add_views(report,figs,ratio,'A02_stim_normalized_difference_TFR','TFR: (stim - no-stim) / (stim + no-stim)',f'Computed from unbaselined multitaper power. NO baseline correction. 2-30 Hz; n_cycles=f/2; time-bandwidth={TIME_BANDWIDTH:g}; FFT=True; ITC=False; trial-average=True; decim={DECIM}; display {PLOT_TMIN} to {PLOT_TMAX} s.',posterior,include_mean=True)

    details={'subject':f'sub-{s}','epoch_original_window_s':[-0.5,1.6],'attention_conditions_combined':['cue_onset_right','cue_onset_left'],'method':'multitaper','frequencies_hz':FREQS.tolist(),'frequency_step_hz':1.0,'n_cycles':'frequency / 2','time_bandwidth':TIME_BANDWIDTH,'use_fft':True,'return_itc':False,'average_trials':True,'decim':DECIM,'display_window_s':[PLOT_TMIN,PLOT_TMAX],'condition_baseline_s':list(BASELINE),'condition_baseline_mode':'percent','comparative_baseline':None,'posterior_order':list(POSTERIOR),'posterior_mean_only_for':'(stim-no-stim)/(stim+no-stim)','vlim_method':f'symmetric +/- {ROBUST_PERCENTILE:g}th percentile of absolute displayed TFR values, shared between scalp-layout and enlarged posterior plots for each result','colormap':'RdBu_r','common_good_eeg_sensors':common,'posterior_available':posterior,'vlim':{k:list(v) for k,v in scales.items()}}
    (qc_dir(root,s)/'A02_tfr_analysis.json').write_text(json.dumps(details,indent=2)+'\n',encoding='utf-8')
    report.add_text('TFR analysis details',f'Input: final cleaned cue epochs (-0.5 to +1.6 s); attention-left and attention-right trials combined within each stimulation condition.\nMultitaper power: 2-30 Hz in 1-Hz steps; n_cycles=f/2; time-bandwidth={TIME_BANDWIDTH:g}; FFT=True; ITC=False; average across retained trials=True; decimation={DECIM}.\nStimulation and no-stimulation displays: percent baseline {BASELINE}. Their titles specify combined attention-left/right TFR.\nDifference and normalized ratio: calculated from unbaselined power with NO baseline correction.\nFor each result, the all-channel scalp layout and enlarged posterior panels are plotted from the exact same TFR object with the same time/frequency range, baseline state, RdBu_r colormap, and one shared robust symmetric color scale (+/- {ROBUST_PERCENTILE:g}th percentile of absolute displayed values).\nPosterior order: PO3 | POz | PO4. Posterior mean is plotted only for the normalized ratio. Scalp-layout figure and axes backgrounds are forced to white.','Time-frequency analysis')
    print(f'TFR complete for sub-{s}. Updated PDF: {report.pdf_fname}')

if __name__=='__main__': main()
