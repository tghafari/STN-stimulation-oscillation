"""A02: TFR analysis in scalp-layout, posterior-separate, and posterior-mean views.
Posterior separate panels are ordered anatomically left-to-right: PO3, POz, PO4.
"""
from __future__ import annotations
import argparse,json
import matplotlib.pyplot as plt
import mne
import numpy as np
from pipeline_config import CONDITIONS,qc_dir,resolve_project_root,stage_path,subject_deriv_dir
from all_channel_report import participant_report,figure_dir,fmt_channels
POSTERIOR=('PO3','POz','PO4'); BASELINE=(-0.3,-0.1); FREQS=np.arange(2.,31.,1.); N_CYCLES=FREQS/2.; TIME_BANDWIDTH=2.; DECIM=2; PLOT_TMIN=-0.3; PLOT_TMAX=1.4
def parse_args():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--subject',required=True); p.add_argument('--session',default='01'); p.add_argument('--task',default='SpAtt'); p.add_argument('--run',default='01'); p.add_argument('--platform',choices=['mac','bluebear'],default='mac'); p.add_argument('--project-root',default=None); p.add_argument('--n-jobs',type=int,default=4); return p.parse_args()
def load_epochs(root,s,a):
    d={}
    for c in CONDITIONS:
        p=stage_path(root,s,a.session,a.task,a.run,c,'clean','epo'); ep=mne.read_epochs(p,preload=True); keep=[k for k in ('cue_onset_right','cue_onset_left') if k in ep.event_id]; d[c]=ep[keep] if keep else ep
    return d
def common_good(d):
    s=d['stim']; n=d['no-stim']; return [ch for ch in s.copy().pick('eeg').ch_names if ch in n.ch_names and ch not in s.info['bads'] and ch not in n.info['bads']]
def compute(ep,jobs): return ep.compute_tfr(method='multitaper',freqs=FREQS,n_cycles=N_CYCLES,time_bandwidth=TIME_BANDWIDTH,use_fft=True,return_itc=False,average=True,decim=DECIM,n_jobs=jobs)
def scalp_topo(tfr): return tfr.plot_topo(tmin=PLOT_TMIN,tmax=PLOT_TMAX,baseline=None,mode=None,show=False)
def posterior_separate(tfr,chs,prefix):
    fig,axes=plt.subplots(1,len(chs),figsize=(5*len(chs),4),constrained_layout=True); axes=[axes] if len(chs)==1 else list(axes)
    for ax,ch in zip(axes,chs): tfr.plot(picks=ch,tmin=PLOT_TMIN,tmax=PLOT_TMAX,baseline=None,mode=None,axes=ax,show=False,colorbar=True); ax.set_title(f'{prefix}: {ch}')
    return fig
def posterior_mean(tfr,chs,title):
    roi=tfr.copy().pick(chs); roi.data=roi.data.mean(axis=0,keepdims=True); roi.info=mne.pick_info(roi.info,[0],copy=True); roi.info['chs'][0]['ch_name']='Posterior mean'; roi.info['ch_names'][0]='Posterior mean'
    fig=roi.plot(picks='Posterior mean',tmin=PLOT_TMIN,tmax=PLOT_TMAX,baseline=None,mode=None,show=False,colorbar=True); fig=fig[0] if isinstance(fig,list) else fig; fig.axes[0].set_title(title); return fig
def add_three_views(report,figs,tfr,stem,title,caption,posterior):
    report.add_figure(scalp_topo(tfr),str(figs/f'{stem}_sensor_topography.png'),f'{title}: all sensors separately in scalp layout',caption+' Each common good EEG sensor has its own TFR panel positioned by electrode location.','Time-frequency analysis')
    if posterior:
        report.add_figure(posterior_separate(tfr,posterior,title),str(figs/f'{stem}_posterior_separate.png'),f'{title}: PO3, POz and PO4 separately',caption+f' Posterior panels are ordered PO3 (left), POz (middle), PO4 (right) when available. Available: {fmt_channels(posterior)}.','Time-frequency analysis')
        report.add_figure(posterior_mean(tfr,posterior,f'{title}: posterior mean'),str(figs/f'{stem}_posterior_mean.png'),f'{title}: mean of PO3, POz and PO4',caption+f' Arithmetic sensor mean across: {fmt_channels(posterior)}.','Time-frequency analysis')
def main():
    a=parse_args(); s=a.subject.removeprefix('sub-'); root=resolve_project_root(a.platform,a.project_root); report=participant_report(root,s); figs=figure_dir(root,s); deriv=subject_deriv_dir(root,s)
    epochs=load_epochs(root,s,a); common=common_good(epochs); posterior=[ch for ch in POSTERIOR if ch in common]
    if not common: raise RuntimeError('No common good EEG sensors available.')
    raw={}; display={}
    for c in CONDITIONS:
        raw[c]=compute(epochs[c].copy().pick(common),a.n_jobs); raw[c].save(deriv/f'sub-{s}_ses-{a.session}_task-{a.task}_run-{a.run}_{c}_desc-allchannels_tfr.h5',overwrite=True)
        display[c]=raw[c].copy().apply_baseline(BASELINE,mode='percent')
        add_three_views(report,figs,display[c],f'A02_{c}_TFR',f'{c}: TFR',f'Multitaper 2-30 Hz; n_cycles=f/2; time-bandwidth={TIME_BANDWIDTH:g}; decim={DECIM}; percent baseline {BASELINE}; display {PLOT_TMIN} to {PLOT_TMAX} s.',posterior)
    diff=raw['stim'].copy(); diff.data=raw['stim'].data-raw['no-stim'].data; diff.save(deriv/f'sub-{s}_ses-{a.session}_task-{a.task}_run-{a.run}_stim-minus-no-stim_desc-allchannels_tfr.h5',overwrite=True)
    add_three_views(report,figs,diff,'A02_stim_minus_no_stim_TFR','TFR: stimulation - no stimulation',f'Computed from unbaselined multitaper power. NO baseline correction. 2-30 Hz; n_cycles=f/2; time-bandwidth={TIME_BANDWIDTH:g}; decim={DECIM}.',posterior)
    ratio=raw['stim'].copy(); denom=raw['stim'].data+raw['no-stim'].data; ratio.data=(raw['stim'].data-raw['no-stim'].data)/(denom+np.finfo(float).eps); ratio.save(deriv/f'sub-{s}_ses-{a.session}_task-{a.task}_run-{a.run}_stim-normalized-difference_desc-allchannels_tfr.h5',overwrite=True)
    add_three_views(report,figs,ratio,'A02_stim_normalized_difference_TFR','TFR: (stim - no-stim) / (stim + no-stim)',f'Computed from unbaselined multitaper power. NO baseline correction. 2-30 Hz; n_cycles=f/2; time-bandwidth={TIME_BANDWIDTH:g}; decim={DECIM}.',posterior)
    details={'subject':f'sub-{s}','epoch_original_window_s':[-0.5,1.6],'method':'multitaper','frequencies_hz':FREQS.tolist(),'n_cycles':'frequency / 2','time_bandwidth':TIME_BANDWIDTH,'decim':DECIM,'condition_baseline_s':list(BASELINE),'condition_baseline_mode':'percent','comparative_baseline':None,'posterior_display_order':list(POSTERIOR),'common_good_eeg_sensors':common,'posterior_available':posterior}
    (qc_dir(root,s)/'A02_tfr_analysis.json').write_text(json.dumps(details,indent=2)+'\n',encoding='utf-8')
    report.add_text('TFR analysis details',f'Input: final cleaned cue epochs (-0.5 to +1.6 s), left/right cue trials combined within stimulation condition.\nMultitaper power: 2-30 Hz in 1-Hz steps; n_cycles=f/2; time-bandwidth={TIME_BANDWIDTH:g}; FFT=True; ITC=False; trial-average=True; decimation={DECIM}.\nStim and no-stim displays: percent baseline {BASELINE}.\nStim-no-stim and normalized difference: calculated from unbaselined power; NO baseline correction.\nPosterior separate-panel order: PO3 (left), POz (middle), PO4 (right).','Time-frequency analysis')
    print(f'TFR complete for sub-{s}. Updated PDF: {report.pdf_fname}')
if __name__=='__main__': main()
