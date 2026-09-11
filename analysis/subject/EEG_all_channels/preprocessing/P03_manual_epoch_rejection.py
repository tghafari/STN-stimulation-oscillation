"""P03: manual bad-trial rejection and final stimulation-artifact spectral QC.

Final PSDs use the same Welch settings as the posterior-channel epoch PSD. After
both conditions are cleaned, an additional QC figure compares stim versus no-stim
on the SAME good EEG channels and plots their PSD ratio. This figure is diagnostic
only: it does not remove or correct stimulation artifact. No butterfly ERP is made.
"""
from __future__ import annotations
import argparse, json
import matplotlib.pyplot as plt
import mne
import numpy as np
from pipeline_config import CONDITIONS, qc_dir, resolve_project_root, stage_path
from all_channel_report import participant_report, figure_dir, fmt_channels


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subject',required=True); p.add_argument('--session',default='01')
    p.add_argument('--task',default='SpAtt'); p.add_argument('--run',default='01')
    p.add_argument('--platform',choices=['mac','bluebear'],default='mac')
    p.add_argument('--project-root',default=None); p.add_argument('--n-channels',type=int,default=20)
    return p.parse_args()


def welch_psd(epochs):
    n_fft=min(int(2*epochs.info['sfreq']),len(epochs.times))
    spectrum=epochs.compute_psd(fmin=0.1,fmax=min(100.0,epochs.info['sfreq']/2.0),method='welch',n_fft=n_fft)
    return spectrum,n_fft


def mean_linear_psd(spectrum):
    data=spectrum.get_data()
    if data.ndim != 3:
        raise RuntimeError(f'Expected EpochsSpectrum data with 3 dimensions (epochs, channels, frequencies), got shape {data.shape}.')
    return data.mean(axis=(0,1))


def stimulation_artifact_qc(clean_epochs, report, figs, subject):
    stim=clean_epochs['stim']; nostim=clean_epochs['no-stim']
    common=[ch for ch in stim.copy().pick('eeg').ch_names if ch in nostim.ch_names and ch not in stim.info['bads'] and ch not in nostim.info['bads']]
    if not common:
        raise RuntimeError('No common good EEG channels available for stimulation-artifact PSD QC.')

    stim_spec,_=welch_psd(stim.copy().pick(common)); nostim_spec,_=welch_psd(nostim.copy().pick(common))
    if not np.allclose(stim_spec.freqs,nostim_spec.freqs):
        raise RuntimeError('Stim and no-stim PSD frequency bins do not match.')
    freqs=stim_spec.freqs
    stim_power=mean_linear_psd(stim_spec); nostim_power=mean_linear_psd(nostim_spec)
    eps=np.finfo(float).tiny
    stim_db=10*np.log10(np.maximum(stim_power,eps)); nostim_db=10*np.log10(np.maximum(nostim_power,eps))
    ratio_db=10*np.log10(np.maximum(stim_power,eps)/np.maximum(nostim_power,eps))

    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(10,8),sharex=True)
    ax1.plot(freqs,nostim_db,label='no-stim'); ax1.plot(freqs,stim_db,label='stim')
    ax1.set_ylabel('Mean EEG PSD (dB)'); ax1.set_title(f'sub-{subject}: stimulation artifact QC'); ax1.legend(); ax1.grid(True,alpha=.25)
    ax2.plot(freqs,ratio_db); ax2.axhline(0,linestyle='--',linewidth=1)
    ax2.set_xlabel('Frequency (Hz)'); ax2.set_ylabel('Stim / no-stim (dB)'); ax2.grid(True,alpha=.25)
    max_tick=int(np.floor(freqs.max()/10.0)*10)
    ax2.set_xticks(np.arange(0,max_tick+1,10))
    ax2.set_xlim(0,min(100.0,float(freqs.max())))
    fig.tight_layout()
    report.add_figure(fig,str(figs/'P03_stimulation_artifact_QC.png'),'Stimulation artifact QC: stim versus no-stim PSD',f'Top: mean Welch PSD averaged across epochs and the same {len(common)} good EEG channels. Bottom: 10*log10(stim/no-stim power). X-axis ticks are shown every 10 Hz. Values above 0 dB indicate greater power during stimulation. This is diagnostic only; no artifact correction is applied.','Stimulation artifact QC')
    report.add_text('How to interpret this QC','This comparison is intended to locate frequencies affected by stimulation before ERP/TFR analysis. Each condition is first averaged across epochs and the identical set of good EEG channels. The ratio is then computed from linear Welch power using identical frequency bins. Peaks in the ratio indicate stimulation-associated spectral power and should not automatically be interpreted as neural modulation. No notch filter, ICA rejection, interpolation, or other correction is performed by this QC step.','Stimulation artifact QC')
    return {'common_good_eeg_channels':common,'n_common_good_eeg_channels':len(common),'averaging':'mean across epochs and common good EEG channels before dB conversion','ratio_definition':'10*log10(mean_stim_linear_power / mean_no_stim_linear_power)','x_ticks_hz':list(range(0,max_tick+1,10)),'qc_only_no_correction':True}


def main():
    a=parse_args(); subject=a.subject.removeprefix('sub-'); root=resolve_project_root(a.platform,a.project_root)
    report=participant_report(root,subject); figs=figure_dir(root,subject); audit={'subject':f'sub-{subject}','conditions':{}}
    clean_epochs={}
    report.add_text('P03 method: manual bad-trial rejection','ICA-cleaned cue epochs are visually inspected using all good EEG channels. Previously marked bad electrodes are not displayed. No automatic epoch rejection is applied. Bad channels remain marked rather than interpolated or dropped.','All-channel EEG preprocessing')
    for condition in CONDITIONS:
        infile=stage_path(root,subject,a.session,a.task,a.run,condition,'ica','epo')
        if not infile.exists(): raise FileNotFoundError(f'Missing ICA-cleaned epochs: {infile}')
        epochs=mne.read_epochs(infile,preload=True); before=len(epochs); bad=list(epochs.info['bads'])
        good=[ch for ch in epochs.copy().pick('eeg').ch_names if ch not in bad]
        if not good: raise RuntimeError('No good EEG channels remain.')
        print('\n'+'='*72); print(f'sub-{subject} / {condition}: MANUAL BAD-TRIAL REJECTION'); print(f'Epochs before: {before}'); print(f'Bad channels excluded from display: {bad or "None"}'); print('='*72)
        epochs.plot(picks=good,n_channels=min(a.n_channels,len(good)),block=True,title=f'sub-{subject} {condition}: manual rejection using all good EEG channels')
        after=len(epochs); outfile=stage_path(root,subject,a.session,a.task,a.run,condition,'clean','epo'); epochs.save(outfile,overwrite=True); clean_epochs[condition]=epochs.copy()
        spectrum,n_fft=welch_psd(epochs); fig_psd=spectrum.plot(show=False)
        report.add_figure(fig_psd,str(figs/f'P03_{condition}_final_epoch_PSD.png'),f'{condition}: PSD of final cleaned epochs',f'Welch PSD using the same settings as the posterior-channel epoch PSD. Retained epochs: {after}; manually rejected: {before-after}.','All-channel EEG preprocessing')
        report.add_text(f'{condition}: manual trial-rejection result',f'Epochs before manual inspection: {before}\nEpochs after manual inspection: {after}\nEpochs manually rejected: {before-after}\nBad channels excluded from visual trial QC: {fmt_channels(bad)}\nNumber of good EEG channels inspected: {len(good)}','All-channel EEG preprocessing')
        audit['conditions'][condition]={'input_epochs':str(infile),'final_epochs':str(outfile),'n_epochs_before_manual_rejection':before,'n_epochs_after_manual_rejection':after,'n_epochs_manually_rejected':before-after,'bad_channels_retained_in_info':list(epochs.info['bads']),'good_eeg_channels_used_for_visual_trial_qc':good,'final_psd':{'method':'welch','fmin_hz':0.1,'fmax_hz':min(100.0,epochs.info['sfreq']/2.0),'n_fft':n_fft}}

    audit['stimulation_artifact_qc']=stimulation_artifact_qc(clean_epochs,report,figs,subject)
    audit_file=qc_dir(root,subject)/'P03_manual_epoch_rejection.json'; audit_file.write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    report.add_text('Preprocessing complete','All-channel preprocessing completed through manual bad-trial rejection. A final stim/no-stim spectral comparison was added for stimulation-artifact QC; it does not alter the EEG data.','All-channel EEG preprocessing')
    print(f'\nUpdated PDF: {report.pdf_fname}')


if __name__=='__main__': main()
