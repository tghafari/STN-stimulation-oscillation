#!/usr/bin/env python
"""Final group report with raw participant/channel contrasts.

TFR logic is fixed and explicit:
1) compute Stim and No-stim power separately per participant and sensor;
2) for that SAME participant/sensor, from ORIGINAL UNBASELINED power compute
   Difference = Stim - No-stim and Ratio = (Stim-No-stim)/(Stim+No-stim);
3) average participant differences/ratios independently for each sensor;
4) only then average channel-level group results across ROI3 or ROI8.

Percent baseline (-0.3,-0.1 s) is used ONLY to display Stim and No-stim TFRs.
It is NEVER used to calculate Difference or Ratio.
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import G04_complete_grand_average_report as b
from pdf_report import ParticipantPDF
from scalp_layout import anatomical_tfr_scalp


def grand_arrays(items):
    times=np.asarray(items[0].times).copy();arr=[]
    for x in items:
        if not np.allclose(x.times,times):raise ValueError('ERP time axes differ')
        arr.append(np.asarray(x.data).squeeze())
    return np.mean(arr,axis=0),times


def roi_erp(no,st,times,title):
    no=np.asarray(no).squeeze();st=np.asarray(st).squeeze();times=np.asarray(times).squeeze()
    fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True);ax.plot(times,no*1e6,label='No stimulation');ax.plot(times,st*1e6,label='Stimulation');ax.axvline(0,color='k',ls='--',lw=.7);ax.set_xlim(b.ERP_TMIN,b.ERP_TMAX);ax.set_xlabel('Time (s)');ax.set_ylabel('Amplitude (uV)');ax.set_title(title);ax.legend();return fig


def nonoverlap_scalp(data,channels,info,title,vlim):
    """Montage-informed discrete layout: anatomical order, guaranteed no overlap."""
    montage=info.get_montage();pos=montage.get_positions().get('ch_pos',{}) if montage else {}
    valid=[ch for ch in channels if ch in pos]
    if len(valid)!=len(channels):
        missing=[ch for ch in channels if ch not in pos];raise RuntimeError('Missing montage positions: '+', '.join(missing))
    xy={ch:np.asarray(pos[ch],float)[:2] for ch in channels};xs=np.array([xy[c][0] for c in channels]);ys=np.array([xy[c][1] for c in channels])
    ncols=11;nrows=11;free={(r,c) for r in range(nrows) for c in range(ncols)};cells={};xmin,xmax=xs.min(),xs.max();ymin,ymax=ys.min(),ys.max()
    # Assign nearest FREE grid cell to true montage target. This preserves head topology
    # while guaranteeing that two channels can never occupy/overlap the same panel.
    targets={ch:((ymax-xy[ch][1])/max(ymax-ymin,1e-12)*(nrows-1),(xy[ch][0]-xmin)/max(xmax-xmin,1e-12)*(ncols-1)) for ch in channels}
    for ch in sorted(channels,key=lambda c:(targets[c][0],targets[c][1])):
        tr,tc=targets[ch];cell=min(free,key=lambda rc:(rc[0]-tr)**2+(rc[1]-tc)**2);cells[ch]=cell;free.remove(cell)
    fig,axs=plt.subplots(nrows,ncols,figsize=(25,23),squeeze=False)
    for ax in axs.ravel():ax.axis('off')
    for ch in channels:
        r,c=cells[ch];ax=axs[r,c];ax.axis('on');arr,t=data[ch];ax.imshow(arr,origin='lower',aspect='auto',extent=[t[0],t[-1],b.FREQS[0],b.FREQS[-1]],cmap='RdBu_r',vmin=vlim[0] if vlim else None,vmax=vlim[1] if vlim else None);ax.axvline(0,color='k',ls='--',lw=.4);ax.set_title(ch,fontsize=8);ax.tick_params(labelsize=5,length=2)
    # Reserve right margin before creating colorbar: it never occupies a sensor cell.
    fig.subplots_adjust(left=.025,right=.89,bottom=.03,top=.94,wspace=.55,hspace=.65);fig.suptitle(title,fontsize=16)
    if vlim:
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        cax=fig.add_axes([.925,.20,.015,.60]);sm=ScalarMappable(norm=Normalize(*vlim),cmap='RdBu_r');sm.set_array([]);fig.colorbar(sm,cax=cax,label='Value')
    return fig


def main():
    a=b.parse_args();subjects=[s.removeprefix('sub-') for s in a.subjects];root=b.resolve_project_root(a.platform,a.project_root)
    out=root/'derivatives'/'reports'/'group'/'EEG_all_channels_complete_grand_average';figs=out/'figures';figs.mkdir(parents=True,exist_ok=True);deriv=root/'data'/'BIDS'/'derivatives'/'group'/'EEG_all_channels_complete_grand_average';deriv.mkdir(parents=True,exist_ok=True)
    rid='complete_grand_average_'+'_'.join(subjects);report=ParticipantPDF(str(out),rid)
    epochs={s:b.load_epochs(root,s,a) for s in subjects};goods={s:b.good_channels(epochs[s]) for s in subjects};channels=[]
    for s in subjects:
        for ch in goods[s]:
            if ch not in channels:channels.append(ch)
    by={ch:[s for s in subjects if ch in goods[s]] for ch in channels};info=epochs[subjects[0]]['stim'].copy().pick('eeg').info
    report.add_text('Subjects included',f'n={len(subjects)}\n'+', '.join('sub-'+s for s in subjects),'Group overview');report.add_text('Subjects contributing to each EEG channel','\n'.join(f"{ch} (n={len(by[ch])}): "+', '.join('sub-'+s for s in by[ch]) for ch in channels),'Group overview')
    logic='TFR contrasts use this exact order: (1) Stim and No-stim power separately for each participant and sensor; (2) from ORIGINAL UNBASELINED power calculate Difference = Stim-No-stim and Ratio = (Stim-No-stim)/(Stim+No-stim) for that participant/sensor; (3) average participant contrasts independently per sensor; (4) average the resulting channel-level group contrasts across PO3/POz/PO4 (ROI3) or PO3/POz/PO4/O1/Oz/O2/PO7/PO8 (ROI8). Percent baseline -0.3 to -0.1 s is used ONLY for descriptive Stim and No-stim plots and never enters Difference or Ratio.'
    report.add_text('Final TFR analysis logic',logic,'Group overview')

    # ERP unchanged.
    subj={s:{c:{} for c in b.CONDITIONS} for s in subjects};grp={c:{} for c in b.CONDITIONS}
    for s in subjects:
        for c in b.CONDITIONS:
            for ch in goods[s]:subj[s][c][ch]=b.make_evoked(epochs[s][c],ch)
    for c in b.CONDITIONS:
        for ch in channels:
            items=[subj[s][c][ch] for s in subjects if ch in subj[s][c]]
            if items:grp[c][ch]=(*grand_arrays(items),len(items))
    report.add_text('ERP analysis details','Cue-locked ERP: trial average per participant/channel, low-pass 30 Hz, baseline -0.1 to 0 s, then participant average per channel. ERP baseline correction is separate from the TFR contrast analysis.','ERP')
    for roi,name in ((b.ROI3,'PO3/POz/PO4'),(b.ROI8,'8 posterior channels')):
        av=[ch for ch in roi if ch in grp['stim'] and ch in grp['no-stim']]
        if av:
            no=np.mean([grp['no-stim'][ch][0] for ch in av],0);st=np.mean([grp['stim'][ch][0] for ch in av],0);report.add_figure(roi_erp(no,st,grp['stim'][av[0]][1],'ERP: mean '+name),str(figs/f'ERP_mean_{len(av)}.png'),'ERP: mean '+name,'Mean channel-level grand-average ERPs.','ERP')

    # TFR power per participant and sensor.
    tfr={s:{c:{} for c in b.CONDITIONS} for s in subjects}
    for s in subjects:
        for c in b.CONDITIONS:
            print(f'Computing TFRs for sub-{s} {c}')
            for ch in goods[s]:tfr[s][c][ch]=b.make_tfr(epochs[s][c],ch,a.n_jobs)
    first=next(tfr[s]['stim'][ch] for s in subjects for ch in tfr[s]['stim']);times=first.times
    display={'stim':{},'no-stim':{}};diff={};ratio={}
    for ch in channels:
        eligible=[s for s in subjects if ch in tfr[s]['stim'] and ch in tfr[s]['no-stim']]
        if not eligible:continue
        # Baseline correction exists ONLY on these two descriptive display arrays.
        display['stim'][ch]=(np.mean([b.baseline_percent_subject_tfr(tfr[s]['stim'][ch])[0] for s in eligible],0),eligible)
        display['no-stim'][ch]=(np.mean([b.baseline_percent_subject_tfr(tfr[s]['no-stim'][ch])[0] for s in eligible],0),eligible)
        # CRITICAL: both contrasts are formed from raw, unbaselined participant/channel power FIRST.
        subject_diff=[tfr[s]['stim'][ch].data[0]-tfr[s]['no-stim'][ch].data[0] for s in eligible]
        subject_ratio=[b.ratio_array(tfr[s]['stim'][ch].data[0],tfr[s]['no-stim'][ch].data[0]) for s in eligible]
        diff[ch]=(np.mean(subject_diff,0),eligible);ratio[ch]=(np.mean(subject_ratio,0),eligible)
    results=[('no-stim','No stimulation',display['no-stim'],True),('stim','Stimulation',display['stim'],True),('difference','Stimulation - no stimulation',diff,False),('ratio','(Stimulation - no stimulation)/(Stimulation + no stimulation)',ratio,False)]
    common='Multitaper TFR 2-31.5 Hz, 0.5-Hz steps, n_cycles=f/2, time-bandwidth=2, FFT=True, zero_mean=True, ITC=False, trial-average=True, decim=2; cue-left/right combined.'
    for key,title,d,is_display_baselined in results:
        data={ch:(v[0],times) for ch,v in d.items()};scale=b.robust_scale([v[0] for v in data.values()]);section='TFR - '+title
        detail=common+(' Percent baseline -0.3 to -0.1 s is applied to this descriptive condition plot.' if is_display_baselined else ' NO baseline correction is applied. The contrast is calculated per participant/sensor from original power, then participants are averaged per sensor.')
        report.add_text('Analysis details',detail,section);report.add_figure(nonoverlap_scalp(data,list(data),info,'Grand-average TFR: '+title,scale),str(figs/f'TFR_{key}_all_channels_scalp.png'),title+': all channels','Montage-informed, non-overlapping scalp layout. Colorbar is outside all sensor panels. '+detail,section)
        pfig,axs=plt.subplots(2,4,figsize=(18,8),constrained_layout=True)
        for ax,ch in zip(axs.ravel(),b.ROI8):
            if ch not in data:ax.axis('off');continue
            arr,t=data[ch];ax.imshow(arr,origin='lower',aspect='auto',extent=[t[0],t[-1],b.FREQS[0],b.FREQS[-1]],cmap='RdBu_r',vmin=scale[0] if scale else None,vmax=scale[1] if scale else None);ax.axvline(0,color='k',ls='--',lw=.5);ax.set_title(f'{ch} (n={len(d[ch][1])})')
        pfig.suptitle(title+': eight posterior channels');report.add_figure(pfig,str(figs/f'TFR_{key}_8posterior.png'),title+': eight posterior channels',detail,section)
        for roi,name,n in ((b.ROI3,'PO3/POz/PO4',3),(b.ROI8,'8 posterior channels',8)):
            arr,av=b.channel_roi(d,roi)
            if arr is not None:
                rs=b.robust_scale([arr]);report.add_figure(b.roi_tfr(arr,times,title+': mean '+name,rs),str(figs/f'TFR_{key}_mean{n}.png'),title+': mean '+name,f'Last step only: arithmetic mean of channel-level group results across {", ".join(av)}. '+detail,section)
    manuscript=f'Final cleaned cue-locked EEG epochs from {len(subjects)} participants were analysed. Time-frequency power was estimated separately for each retained EEG sensor and stimulation condition. For each participant and sensor, two contrasts were calculated directly from original unbaselined power: the absolute difference P_stim - P_no-stim and the normalized difference (P_stim - P_no-stim)/(P_stim + P_no-stim). These participant-level sensor contrasts were then averaged across eligible participants independently for each sensor. Posterior ROI summaries were calculated only after sensor-level group averaging, as the arithmetic mean across PO3, POz and PO4 for ROI3 and across PO3, POz, PO4, O1, Oz, O2, PO7 and PO8 for ROI8. Thus both contrasts followed: power per participant/sensor -> contrast per participant/sensor -> average participants per sensor -> average sensors. No baseline correction was applied to either difference or ratio. Percent baseline correction from -0.3 to -0.1 s was used only for descriptive Stim and No-stim TFR plots and did not enter either contrast.'
    report.add_text('Analysis report - manuscript style',manuscript,'Analysis');report.add_text('Exact reproducibility parameters',logic+'\nDifference baseline: NONE.\nRatio baseline: NONE.\nStim/No-stim descriptive baseline: percent, -0.3 to -0.1 s.','Analysis')
    (deriv/f'{rid}_analysis.json').write_text(json.dumps({'subjects':subjects,'subjects_by_channel':by,'contrast_order':'raw power per participant/sensor -> difference and ratio per participant/sensor -> mean participants per sensor -> mean sensors','difference_baseline':'none','ratio_baseline':'none','stim_nostim_display_baseline':'percent -0.3 to -0.1 s'},indent=2)+'\n')
    print(f'Complete report: {report.pdf_fname}')

if __name__=='__main__':main()
