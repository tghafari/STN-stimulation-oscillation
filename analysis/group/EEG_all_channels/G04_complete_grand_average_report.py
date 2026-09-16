#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Final all-channel cue-locked EEG grand-average report.

FINAL NORMALIZED TFR LOGIC
--------------------------
For each participant and each EEG channel retained in BOTH stimulation conditions:
    Pstim(ch, subject) = cue-locked TFR power
    Pnostim(ch, subject) = cue-locked TFR power
    R(ch, subject) = (Pstim - Pnostim) / (Pstim + Pnostim)

Then, independently for every EEG channel:
    R_group(ch) = mean_subjects[R(ch, subject)]

Posterior ROI summaries are then formed from the channel-level group results:
    ROI3 = mean[R_group(PO3), R_group(POz), R_group(PO4)]
    ROI8 = mean[R_group(PO3), R_group(POz), R_group(O1), R_group(Oz),
                R_group(O2), R_group(PO4), R_group(PO7), R_group(PO8)]

Therefore the final ROI ratio is:
    subject/channel power -> subject/channel ratio -> mean subjects per channel
    -> mean channels within ROI.

There is NO EEG-domain ROI averaging before TFR, NO ratio of group means, and NO
mean of participant ROI ratios in this final analysis.

Descriptive stimulation, no-stimulation and difference TFR ROI summaries follow the
same channel-first group structure: group-average each channel first, then average
channel results within the requested ROI.

All report files are written only to:
    derivatives/reports/group/EEG_all_channels_complete_grand_average/
"""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import mne
import numpy as np
HERE=Path(__file__).resolve().parent;ANALYSIS_DIR=HERE.parents[1]
for p in (ANALYSIS_DIR/'subject'/'EEG_all_channels',ANALYSIS_DIR/'utils'):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
from pipeline_config import CONDITIONS,resolve_project_root,stage_path
from pdf_report import ParticipantPDF

ROI3=('PO3','POz','PO4')
ROI8=('PO3','POz','PO4','O1','Oz','O2','PO7','PO8')
ERP_BASELINE=(-.1,0.);ERP_LP=30.;ERP_TMIN=-.1;ERP_TMAX=1.0
BASELINE=(-.3,-.1);FREQS=np.arange(2.,32.,.5);N_CYCLES=FREQS/2.;TIME_BANDWIDTH=2.;DECIM=2
TFR_TMIN=-.5;TFR_TMAX=1.5;ROBUST=98.

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subjects',nargs='+',required=True)
    p.add_argument('--session',default='01');p.add_argument('--task',default='SpAtt');p.add_argument('--run',default='01')
    p.add_argument('--platform',choices=['mac','bluebear'],default='mac');p.add_argument('--project-root',default=None);p.add_argument('--n-jobs',type=int,default=4)
    return p.parse_args()

def load_epochs(root,s,a):
    out={}
    for c in CONDITIONS:
        f=stage_path(root,s,a.session,a.task,a.run,c,'clean','epo')
        if not f.exists():raise FileNotFoundError(f'Missing final cleaned epochs for sub-{s} {c}: {f}')
        e=mne.read_epochs(f,preload=True,verbose=False)
        keys=[k for k in ('cue_onset_right','cue_onset_left') if k in e.event_id]
        out[c]=e[keys] if keys else e
    return out

def good_channels(pair):
    st,no=pair['stim'],pair['no-stim']
    return [ch for ch in st.copy().pick('eeg').ch_names if ch in no.ch_names and ch not in st.info['bads'] and ch not in no.info['bads']]

def make_tfr(ep,ch,jobs):
    return ep.copy().pick([ch]).compute_tfr(method='multitaper',freqs=FREQS,n_cycles=N_CYCLES,time_bandwidth=TIME_BANDWIDTH,use_fft=True,zero_mean=True,return_itc=False,average=True,decim=DECIM,n_jobs=jobs,verbose=False)

def make_evoked(ep,ch):
    e=ep.copy().pick([ch]).average();e.filter(None,ERP_LP,verbose=False);e.apply_baseline(ERP_BASELINE);e.crop(ERP_TMIN,ERP_TMAX);return e

def ratio_array(st,no):
    return (st-no)/(st+no+np.finfo(float).eps)

def baseline_percent_subject_tfr(tfr):
    x=tfr.copy();mask=(x.times>=BASELINE[0])&(x.times<=BASELINE[1]);base=np.mean(x.data[:,:,mask],axis=2,keepdims=True);return 100*(x.data-base)/np.where(np.abs(base)<np.finfo(float).eps,1,base)

def robust_scale(arrays):
    vals=[]
    for a in arrays:
        if a is None:continue
        x=np.asarray(a);x=x[np.isfinite(x)]
        if x.size:vals.append(x)
    if not vals:return None
    v=np.concatenate(vals);m=float(np.percentile(np.abs(v),ROBUST));return(-m,m) if m>0 else None

def grid_cells(channels,info,ncols=9):
    montage=info.get_montage();pos=montage.get_positions().get('ch_pos',{}) if montage is not None else {}
    coords={ch:(np.asarray(pos[ch],float)[:2] if ch in pos else np.array([i%ncols,-(i//ncols)],float)) for i,ch in enumerate(channels)}
    xs=np.array([coords[c][0] for c in channels]);ys=np.array([coords[c][1] for c in channels]);nrows=int(np.ceil(len(channels)/ncols));free={(r,c) for r in range(nrows) for c in range(ncols)};cells={}
    xmin,xmax=xs.min(),xs.max();ymin,ymax=ys.min(),ys.max()
    for ch in sorted(channels,key=lambda c:coords[c][1],reverse=True):
        x,y=coords[ch];tc=int(round((x-xmin)/max(xmax-xmin,1e-9)*(ncols-1)));tr=int(round((ymax-y)/max(ymax-ymin,1e-9)*(nrows-1)));cell=min(free,key=lambda rc:(rc[0]-tr)**2+(rc[1]-tc)**2);cells[ch]=cell;free.remove(cell)
    return cells,nrows

def scalp_erp(group_erp,channels,info):
    cells,nrows=grid_cells(channels,info);fig,axs=plt.subplots(nrows,9,figsize=(24,3.1*nrows),squeeze=False)
    for ax in axs.ravel():ax.axis('off')
    for ch in channels:
        if ch not in group_erp['stim'] or ch not in group_erp['no-stim']:continue
        r,c=cells[ch];ax=axs[r,c];ax.axis('on');times=group_erp['stim'][ch][1];ax.plot(times,group_erp['no-stim'][ch][0]*1e6,label='No stimulation',lw=.9);ax.plot(times,group_erp['stim'][ch][0]*1e6,label='Stimulation',lw=.9);ax.axvline(0,color='k',ls='--',lw=.5);ax.set_title(f'{ch} (n={len(group_erp["stim"][ch][1])})',fontsize=8);ax.tick_params(labelsize=5);ax.set_xlim(ERP_TMIN,ERP_TMAX)
    handles,labels=next(ax for ax in axs.ravel() if ax.has_data()).get_legend_handles_labels();fig.legend(handles,labels,loc='upper right');fig.suptitle('Grand-average cue-locked ERP: all channels',fontsize=15);fig.subplots_adjust(left=.03,right=.96,bottom=.04,top=.93,wspace=.55,hspace=.75);return fig

def tfr_scalp(data,channels,info,title,vlim):
    cells,nrows=grid_cells(channels,info);fig,axs=plt.subplots(nrows,9,figsize=(22,3.0*nrows),squeeze=False)
    for ax in axs.ravel():ax.axis('off')
    for ch in channels:
        if ch not in data:continue
        r,c=cells[ch];ax=axs[r,c];ax.axis('on');arr,t=data[ch];ax.imshow(arr,origin='lower',aspect='auto',extent=[t[0],t[-1],FREQS[0],FREQS[-1]],cmap='RdBu_r',vmin=vlim[0] if vlim else None,vmax=vlim[1] if vlim else None);ax.axvline(0,color='k',ls='--',lw=.5);ax.set_title(ch,fontsize=8);ax.tick_params(labelsize=5)
    if vlim:
        sm=ScalarMappable(norm=Normalize(vlim[0],vlim[1]),cmap='RdBu_r');sm.set_array([]);fig.colorbar(sm,ax=axs.ravel().tolist(),shrink=.65,label='Value')
    fig.suptitle(title,fontsize=15);fig.subplots_adjust(left=.03,right=.94,bottom=.04,top=.92,wspace=.6,hspace=.75);return fig

def posterior_8(data,channels,title,vlim):
    fig,axs=plt.subplots(2,4,figsize=(18,8),constrained_layout=True)
    for ax,ch in zip(axs.ravel(),ROI8):
        if ch not in data:ax.axis('off');ax.set_title(f'{ch}: unavailable');continue
        arr,t=data[ch];ax.imshow(arr,origin='lower',aspect='auto',extent=[t[0],t[-1],FREQS[0],FREQS[-1]],cmap='RdBu_r',vmin=vlim[0] if vlim else None,vmax=vlim[1] if vlim else None);ax.axvline(0,color='k',ls='--',lw=.5);ax.set_title(ch)
    fig.suptitle(title);return fig

def roi_tfr(arr,times,title,vlim):
    fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True);im=ax.imshow(arr,origin='lower',aspect='auto',extent=[times[0],times[-1],FREQS[0],FREQS[-1]],cmap='RdBu_r',vmin=vlim[0] if vlim else None,vmax=vlim[1] if vlim else None);ax.axvline(0,color='k',ls='--',lw=.7);ax.set_xlabel('Time (s)');ax.set_ylabel('Frequency (Hz)');ax.set_title(title);fig.colorbar(im,ax=ax,label='Value');return fig

def roi_erp(no,st,times,title):
    fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True);ax.plot(times,no*1e6,label='No stimulation');ax.plot(times,st*1e6,label='Stimulation');ax.axvline(0,color='k',ls='--',lw=.7);ax.set_xlim(ERP_TMIN,ERP_TMAX);ax.set_xlabel('Time (s)');ax.set_ylabel('Amplitude (uV)');ax.set_title(title);ax.legend();return fig

def channel_roi(data_by_channel,roi):
    av=[ch for ch in roi if ch in data_by_channel]
    if not av:return None,[]
    return np.mean([data_by_channel[ch][0] for ch in av],axis=0),av

def main():
    a=parse_args();subjects=[s.removeprefix('sub-') for s in a.subjects];root=resolve_project_root(a.platform,a.project_root)
    out=root/'derivatives'/'reports'/'group'/'EEG_all_channels_complete_grand_average';figs=out/'figures';figs.mkdir(parents=True,exist_ok=True)
    deriv=root/'data'/'BIDS'/'derivatives'/'group'/'EEG_all_channels_complete_grand_average';deriv.mkdir(parents=True,exist_ok=True)
    rid='complete_grand_average_'+'_'.join(subjects);report=ParticipantPDF(str(out),rid)
    epochs={s:load_epochs(root,s,a) for s in subjects};goods={s:good_channels(epochs[s]) for s in subjects};channels=[]
    for s in subjects:
        for ch in goods[s]:
            if ch not in channels:channels.append(ch)
    by={ch:[s for s in subjects if ch in goods[s]] for ch in channels};info=epochs[subjects[0]]['stim'].copy().pick('eeg').info
    report.add_text('Subjects included',f'n={len(subjects)}\n'+', '.join('sub-'+s for s in subjects),'Group overview')
    report.add_text('Subjects contributing to each EEG channel','\n'.join(f"{ch} (n={len(by[ch])}): "+', '.join('sub-'+s for s in by[ch]) for ch in channels),'Group overview')
    report.add_text('Final analysis logic','For each participant and each retained EEG channel, stimulation and no-stimulation TFR power are calculated separately. The normalized ratio is then calculated per participant and channel: (stim - no-stim)/(stim + no-stim). For each channel, these participant-level ratios are averaged across eligible subjects. Finally, the 3-channel posterior ROI is the mean of the PO3, POz and PO4 channel-level group ratios; the 8-channel ROI is the mean of the eight posterior channel-level group ratios. Thus the order is power per channel per subject -> ratio per channel per subject -> mean subjects per channel -> mean channels. There is no ratio of group means and no participant-level ROI averaging.','Group overview')

    # ERP: subject -> channel average -> group channel average -> ROI channel average.
    subj_erp={s:{c:{} for c in CONDITIONS} for s in subjects};group_erp={c:{} for c in CONDITIONS}
    for s in subjects:
        for c in CONDITIONS:
            for ch in goods[s]:subj_erp[s][c][ch]=make_evoked(epochs[s][c],ch)
    for c in CONDITIONS:
        for ch in channels:
            items=[subj_erp[s][c][ch] for s in subjects if ch in subj_erp[s][c]]
            if items:
                dat,t=grand_arrays(items);group_erp[c][ch]=(dat,t, len(items))
    erp_data={ch:(group_erp['stim'][ch][0],group_erp['stim'][ch][1],group_erp['no-stim'][ch][0],group_erp['no-stim'][ch][1]) for ch in channels if ch in group_erp['stim'] and ch in group_erp['no-stim']}
    report.add_text('ERP analysis details','Cue-locked ERP only. For every participant and channel, trials are averaged; ERPs are low-pass filtered at 30 Hz and baseline corrected -0.1 to 0 s. Participant channel ERPs are then grand-averaged independently per sensor. Posterior ROI ERPs are calculated by averaging these channel-level grand-average ERPs across the requested channels.','ERP')
    # Large scalp layout and posterior 2x4.
    cells,nrows=grid_cells(list(erp_data),info);fig,axs=plt.subplots(nrows,9,figsize=(24,3.1*nrows),squeeze=False)
    for ax in axs.ravel():ax.axis('off')
    for ch in erp_data:
        r,c=cells[ch];ax=axs[r,c];ax.axis('on');st,ts,no,tn=erp_data[ch];ax.plot(ts,no*1e6,label='No stimulation',lw=.9);ax.plot(ts,st*1e6,label='Stimulation',lw=.9);ax.axvline(0,color='k',ls='--',lw=.5);ax.set_title(f'{ch} (n={len(by[ch])})',fontsize=8);ax.tick_params(labelsize=5);ax.set_xlim(ERP_TMIN,ERP_TMAX)
    handles,labels=next(ax for ax in axs.ravel() if ax.has_data()).get_legend_handles_labels();fig.legend(handles,labels,loc='upper right');fig.suptitle('Grand-average cue-locked ERP: all channels');fig.subplots_adjust(left=.03,right=.96,bottom=.04,top=.93,wspace=.55,hspace=.75);report.add_figure(fig,str(figs/'ERP_all_channels_scalp.png'),'ERP: all channels','Large readable non-overlapping scalp-like layout.','ERP')
    pfig,axs=plt.subplots(2,4,figsize=(18,8),constrained_layout=True)
    for ax,ch in zip(axs.ravel(),ROI8):
        if ch not in erp_data:ax.axis('off');ax.set_title(f'{ch}: unavailable');continue
        st,ts,no,tn=erp_data[ch];ax.plot(ts,no*1e6,label='No stimulation');ax.plot(ts,st*1e6,label='Stimulation');ax.axvline(0,color='k',ls='--',lw=.6);ax.set_xlim(ERP_TMIN,ERP_TMAX);ax.set_title(f'{ch} (n={len(by[ch])})')
    axs[0,0].legend();pfig.suptitle('Grand-average ERP: eight posterior channels (2 x 4)');report.add_figure(pfig,str(figs/'ERP_8posterior.png'),'ERP: eight posterior channels','PO3, POz, PO4, O1, Oz, O2, PO7, PO8.','ERP')
    for roi,name in ((ROI3,'PO3/POz/PO4'),(ROI8,'8 posterior channels')):
        av=[ch for ch in roi if ch in erp_data];
        if av:
            no=np.mean([group_erp['no-stim'][ch][0] for ch in av],axis=0);st=np.mean([group_erp['stim'][ch][0] for ch in av],axis=0);report.add_figure(roi_erp(no[None,:],st[None,:],next(iter(erp_data.values()))[1],'ERP: mean '+name),str(figs/f'ERP_mean_{len(av)}.png'),'ERP: mean '+name,'Mean of channel-level grand-average ERPs.','ERP')

    # TFRs per participant/channel.
    subject_tfr={s:{c:{} for c in CONDITIONS} for s in subjects}
    for s in subjects:
        for c in CONDITIONS:
            print(f'Computing TFRs for sub-{s} {c}')
            for ch in goods[s]:subject_tfr[s][c][ch]=make_tfr(epochs[s][c],ch,a.n_jobs)
    times=next(iter(subject_tfr[subjects[0]]['stim'].values())).times
    group_power={'stim':{},'no-stim':{}};group_ratio={};
    for ch in channels:
        eligible=[s for s in subjects if ch in subject_tfr[s]['stim'] and ch in subject_tfr[s]['no-stim']]
        if not eligible:continue
        group_power['stim'][ch]=(np.mean([baseline_percent_subject_tfr(subject_tfr[s]['stim'][ch])[0] for s in eligible],axis=0),eligible)
        group_power['no-stim'][ch]=(np.mean([baseline_percent_subject_tfr(subject_tfr[s]['no-stim'][ch])[0] for s in eligible],axis=0),eligible)
        subject_ratios=[ratio_array(subject_tfr[s]['stim'][ch].data[0],subject_tfr[s]['no-stim'][ch].data[0]) for s in eligible]
        group_ratio[ch]=(np.mean(subject_ratios,axis=0),eligible)
    # Raw difference of channel-level group power, while stim/no-stim display is percent baseline.
    group_diff={ch:(group_power['stim'][ch][0]-group_power['no-stim'][ch][0],group_power['stim'][ch][1]) for ch in group_power['stim'] if ch in group_power['no-stim']}
    result_defs=[('no-stim','No stimulation',group_power['no-stim']),('stim','Stimulation',group_power['stim']),('difference','Stimulation - no stimulation',group_diff),('ratio','(Stimulation - no stimulation)/(Stimulation + no stimulation)',group_ratio)]
    common='TFR: multitaper, 2-31.5 Hz in 0.5-Hz steps, n_cycles=f/2, time-bandwidth=2, FFT=True, zero_mean=True, ITC=False, average=True, decim=2, cue-left/right combined.'
    for key,title,d in result_defs:
        data={ch:(v[0],times) for ch,v in d.items()};scale=robust_scale([v[0] for v in data.values()]);section='TFR - '+title
        detail=common
        if key in ('no-stim','stim'):detail+=f' Display uses participant-level percent baseline -0.3 to -0.1 s before channel-level group averaging.'
        elif key=='difference':detail+=' Difference is calculated from the channel-level group-average displayed TFRs.'
        else:detail+=' Ratio is calculated independently for every participant and channel from original unbaselined powers, then averaged across subjects for each channel.'
        report.add_text('Analysis details',detail+f' Final posterior ROI values are arithmetic means across channel-level group results. Scale={scale}.',section)
        report.add_figure(tfr_scalp(data,list(data),info,'Grand-average TFR: '+title,scale),str(figs/f'TFR_{key}_all_channels_scalp.png'),title+': all channels','Readable non-overlapping scalp-like layout. '+detail,section)
        pfig,axs=plt.subplots(2,4,figsize=(18,8),constrained_layout=True)
        for ax,ch in zip(axs.ravel(),ROI8):
            if ch not in data:ax.axis('off');ax.set_title(f'{ch}: unavailable');continue
            arr,t=data[ch];ax.imshow(arr,origin='lower',aspect='auto',extent=[t[0],t[-1],FREQS[0],FREQS[-1]],cmap='RdBu_r',vmin=scale[0] if scale else None,vmax=scale[1] if scale else None);ax.axvline(0,color='k',ls='--',lw=.5);ax.set_title(f'{ch} (n={len(by[ch])})')
        pfig.suptitle(title+': eight posterior channels');report.add_figure(pfig,str(figs/f'TFR_{key}_8posterior.png'),title+': eight posterior channels','PO3, POz, PO4, O1, Oz, O2, PO7, PO8.','TFR')
        for roi,name,n in ((ROI3,'PO3/POz/PO4',3),(ROI8,'8 posterior channels',8)):
            arr,av=channel_roi(d,roi)
            if arr is not None:
                rs=robust_scale([arr]);report.add_figure(roi_tfr(arr,times,title+': mean '+name,rs),str(figs/f'TFR_{key}_mean{n}.png'),title+': mean '+name,f'Final ROI = mean of channel-level group results across {", ".join(av)}. Each channel was first averaged across eligible subjects.','TFR')
    manuscript=f'Final cleaned cue-locked EEG epochs from {len(subjects)} participants were analysed. Cue-left and cue-right epochs were combined within stimulation condition. No epochs were concatenated across participants. For each participant and EEG channel retained in both conditions, stimulation and no-stimulation time-frequency power was estimated separately using multitaper decomposition from 2 to 31.5 Hz in 0.5-Hz steps, with n_cycles equal to frequency/2, time-bandwidth=2, FFT enabled, zero-mean tapers, trial averaging and decimation=2. For every channel and participant, the normalized stimulation contrast was calculated from original unbaselined power as (P_stim - P_no-stim)/(P_stim + P_no-stim). Participant channel ratios were then averaged across eligible subjects independently for each channel. Finally, the three-channel posterior ROI was calculated as the arithmetic mean of the channel-level grand-average ratios for PO3, POz and PO4, and the eight-channel ROI as the arithmetic mean of the channel-level grand-average ratios for PO3, POz, PO4, O1, Oz, O2, PO7 and PO8. Thus the final order was: power per channel per subject -> ratio per channel per subject -> average across subjects per channel -> average across channels. This approach gives equal weight to each channel at the final ROI step; subject counts can differ between channels. It is distinct from computing an ROI in the EEG time domain before TFR estimation, taking a ratio of group means, or averaging participant-level ROI ratios.'
    report.add_text('Analysis report - manuscript style',manuscript,'Analysis')
    report.add_text('Exact reproducibility parameters',f'Subjects: '+', '.join('sub-'+s for s in subjects)+f'\nROI3={ROI3}\nROI8={ROI8}\nRatio order: subject/channel power -> subject/channel ratio -> mean subjects per channel -> mean channels within ROI.\nRatio baseline correction: none. Stim/no-stim display baseline={BASELINE}.\nAll report files: {out}','Analysis')
    (deriv/f'{rid}_analysis.json').write_text(json.dumps({'subjects':['sub-'+s for s in subjects],'subjects_by_channel':by,'ROI3':ROI3,'ROI8':ROI8,'order':'power per channel per subject -> ratio per channel per subject -> mean across subjects per channel -> mean across channels','ratio_baseline':'none'},indent=2)+'\n')
    print(f'Complete report: {report.pdf_fname}')

if __name__=='__main__':main()
