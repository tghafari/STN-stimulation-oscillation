"""A01: cue-locked ERP analysis in three complementary sensor views.

For stimulation versus no stimulation this script produces:
1. one scalp-layout figure with a separate ERP trace at every common good EEG sensor;
2. posterior traces ordered anatomically left-to-right: PO3, POz, PO4;
3. one enlarged trace representing the arithmetic mean of available PO3/POz/PO4.
"""
from __future__ import annotations
import argparse, json
import matplotlib.pyplot as plt
import mne
from pipeline_config import CONDITIONS, qc_dir, resolve_project_root, stage_path
from all_channel_report import participant_report, figure_dir, fmt_channels

POSTERIOR=('PO3','POz','PO4')
ERP_BASELINE=(-0.1,0.0); ERP_LP_HZ=30.0; ERP_TMIN=-0.1; ERP_TMAX=0.5

def parse_args():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--subject',required=True); p.add_argument('--session',default='01'); p.add_argument('--task',default='SpAtt'); p.add_argument('--run',default='01'); p.add_argument('--platform',choices=['mac','bluebear'],default='mac'); p.add_argument('--project-root',default=None); return p.parse_args()
def load_clean_epochs(root,subject,a):
    out={}
    for c in CONDITIONS:
        path=stage_path(root,subject,a.session,a.task,a.run,c,'clean','epo')
        if not path.exists(): raise FileNotFoundError(f'Missing final cleaned epochs for {c}: {path}')
        ep=mne.read_epochs(path,preload=True); keep=[k for k in ('cue_onset_right','cue_onset_left') if k in ep.event_id]
        if keep: ep=ep[keep]
        out[c]=ep
    return out
def common_good_eeg(epochs):
    stim=epochs['stim']; nostim=epochs['no-stim']; return [ch for ch in stim.copy().pick('eeg').ch_names if ch in nostim.ch_names and ch not in stim.info['bads'] and ch not in nostim.info['bads']]
def make_evoked(ep,picks):
    ev=ep.copy().pick(picks).average(method='mean'); ev.filter(None,ERP_LP_HZ); ev.apply_baseline(ERP_BASELINE); ev.crop(ERP_TMIN,ERP_TMAX); return ev
def main():
    a=parse_args(); subject=a.subject.removeprefix('sub-'); root=resolve_project_root(a.platform,a.project_root); report=participant_report(root,subject); figs=figure_dir(root,subject)
    epochs=load_clean_epochs(root,subject,a); common=common_good_eeg(epochs)
    if not common: raise RuntimeError('No common good EEG sensors are available for ERP comparison.')
    ev={c:make_evoked(epochs[c],common) for c in CONDITIONS}
    for c in CONDITIONS: mne.write_evokeds(stage_path(root,subject,a.session,a.task,a.run,c,'erp','ave'),ev[c],overwrite=True)
    compare={'no stimulation':ev['no-stim'],'stimulation':ev['stim']}
    fig_topo=mne.viz.plot_compare_evokeds(compare,picks=common,combine=None,axes='topo',show=False,ci=False,truncate_xaxis=False,truncate_yaxis=False,legend=True)
    if isinstance(fig_topo,list): fig_topo=fig_topo[0]
    report.add_figure(fig_topo,str(figs/'A01_ERP_stim_vs_no_stim_sensor_topography.png'),'ERP: stimulation vs no stimulation, separate traces at all available sensors',f'Each of the {len(common)} common good EEG sensors has its own ERP trace and is positioned according to the scalp montage. Stimulation and no-stimulation are overlaid within each sensor. Trial mean; {ERP_LP_HZ:g}-Hz low-pass; baseline {ERP_BASELINE}; display {ERP_TMIN} to {ERP_TMAX} s.','ERP analysis')
    posterior=[ch for ch in POSTERIOR if ch in common]
    if posterior:
        fig_post,axes=plt.subplots(1,len(posterior),figsize=(5*len(posterior),4),constrained_layout=True); axes=[axes] if len(posterior)==1 else list(axes)
        for ax,ch in zip(axes,posterior):
            mne.viz.plot_compare_evokeds(compare,picks=ch,combine=None,axes=ax,show=False,ci=False,truncate_xaxis=False,truncate_yaxis=False); ax.axvline(0,color='k',linestyle='--',linewidth=1); ax.set_xlim(ERP_TMIN,ERP_TMAX); ax.set_title(ch)
        report.add_figure(fig_post,str(figs/'A01_ERP_stim_vs_no_stim_posterior_separate.png'),'ERP: PO3, POz and PO4 separately',f'Posterior sensors are displayed anatomically left-to-right as PO3, POz, PO4 when available. Available: {fmt_channels(posterior)}. Trial mean; {ERP_LP_HZ:g}-Hz low-pass; baseline {ERP_BASELINE}.','ERP analysis')
        fig_mean=mne.viz.plot_compare_evokeds(compare,picks=posterior,combine='mean',show=False,ci=False,truncate_xaxis=False,truncate_yaxis=False)
        if isinstance(fig_mean,list): fig_mean=fig_mean[0]
        ax=fig_mean.axes[0]; ax.axvline(0,color='k',linestyle='--',linewidth=1); ax.set_xlim(ERP_TMIN,ERP_TMAX); ax.set_title(f'sub-{subject}: mean posterior ERP ({", ".join(posterior)})')
        report.add_figure(fig_mean,str(figs/'A01_ERP_stim_vs_no_stim_posterior_mean.png'),'ERP: mean of PO3, POz and PO4',f'Arithmetic sensor mean across the available predefined posterior ROI: {fmt_channels(posterior)}.','ERP analysis')
    else: report.add_text('Posterior ERP unavailable','None of PO3, POz or PO4 was good in both conditions.','ERP analysis')
    details={'subject':f'sub-{subject}','epoch_original_window_s':[-0.5,1.6],'erp_display_window_s':[ERP_TMIN,ERP_TMAX],'baseline_s':list(ERP_BASELINE),'evoked_low_pass_hz':ERP_LP_HZ,'posterior_display_order':list(POSTERIOR),'posterior_available':posterior,'common_good_eeg_sensors':common}
    (qc_dir(root,subject)/'A01_erp_analysis.json').write_text(json.dumps(details,indent=2)+'\n',encoding='utf-8')
    report.add_text('ERP analysis details',f'Input epochs: -0.5 to +1.6 s around cue onset; attention-left/right combined within condition.\nERP: arithmetic mean across retained trials; low-pass {ERP_LP_HZ:g} Hz; baseline {ERP_BASELINE[0]:g} to {ERP_BASELINE[1]:g} s; display {ERP_TMIN:g} to {ERP_TMAX:g} s.\nPosterior separate-panel order: PO3 (left), POz (middle), PO4 (right).','ERP analysis')
    print(f'ERP complete for sub-{subject}. Updated PDF: {report.pdf_fname}')
if __name__=='__main__': main()
