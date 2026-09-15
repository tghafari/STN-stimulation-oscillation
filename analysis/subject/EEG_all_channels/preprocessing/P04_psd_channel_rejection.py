"""P04: optional channel rejection after inspection of the final epoch PSD.

Run this immediately after P03_manual_epoch_rejection.py if the final PSD plots reveal
an additional persistently noisy EEG channel. This step is deliberately interactive.
It never removes a channel automatically.

The user may enter one or more EEG channel names after reviewing the P03 PSD figures.
The selected channels are marked bad in BOTH stimulation conditions, a reason is
recorded for every newly rejected channel, the final clean epoch files are overwritten,
and new PSD plots are generated so the user can verify the result.

If no channel should be removed, press Enter and the clean files are left unchanged.
"""
from __future__ import annotations
import argparse, json
import matplotlib.pyplot as plt
import mne
from pipeline_config import CONDITIONS, qc_dir, resolve_project_root, stage_path
from all_channel_report import participant_report, figure_dir, fmt_channels


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subject',required=True)
    p.add_argument('--session',default='01')
    p.add_argument('--task',default='SpAtt')
    p.add_argument('--run',default='01')
    p.add_argument('--platform',choices=['mac','bluebear'],default='mac')
    p.add_argument('--project-root',default=None)
    return p.parse_args()


def welch(ep):
    n=min(int(2*ep.info['sfreq']),len(ep.times))
    return ep.compute_psd(fmin=0.1,fmax=min(100.,ep.info['sfreq']/2.),method='welch',n_fft=n),n

def main():
    a=parse_args(); s=a.subject.removeprefix('sub-'); root=resolve_project_root(a.platform,a.project_root)
    report=participant_report(root,s); figs=figure_dir(root,s)
    epochs={c:mne.read_epochs(stage_path(root,s,a.session,a.task,a.run,c,'clean','epo'),preload=True) for c in CONDITIONS}
    eeg=epochs['stim'].copy().pick('eeg').ch_names
    existing=sorted(set().union(*(set(epochs[c].info['bads']) for c in CONDITIONS)))

    print('\nReview the P03 final PSD figures before continuing.')
    print('Currently marked bad EEG channels:', ', '.join(existing) if existing else 'none')
    while True:
        add=input('EEG channel(s) to mark BAD based on PSD (space/comma separated; Enter for none): ').replace(',',' ').split()
        unknown=[ch for ch in add if ch not in eeg]
        if not unknown: break
        print('Unknown EEG channels:', unknown)

    new=sorted(set(add)-set(existing))
    reasons={}
    for ch in new:
        while True:
            reason=input(f'Reason for rejecting {ch} based on PSD: ').strip()
            if reason: break
            print('Please enter a reason.')
        reasons[ch]=reason

    final=sorted(set(existing)|set(add))
    audit={'subject':f'sub-{s}','bad_channels_before_psd_review':existing,
           'additional_bad_channels_from_psd':new,'reasons':reasons,
           'final_bad_channels':final,'conditions':{}}

    if not new:
        report.add_text('Post-PSD channel review','PSD reviewed; no additional channels were rejected.','All-channel EEG preprocessing')
        audit['changed_clean_files']=False
        (qc_dir(root,s)/'P04_psd_channel_rejection.json').write_text(json.dumps(audit,indent=2)+'\n')
        print('No additional PSD-based channel rejection. Clean epoch files unchanged.')
        return

    for c in CONDITIONS:
        ep=epochs[c]
        ep.info['bads']=[ch for ch in final if ch in ep.ch_names]
        out=stage_path(root,s,a.session,a.task,a.run,c,'clean','epo')
        ep.save(out,overwrite=True)
        good=[ch for ch in ep.copy().pick('eeg').ch_names if ch not in ep.info['bads']]
        spec,n=welch(ep.copy().pick(good))
        fig=spec.plot(show=False)
        report.add_figure(fig,str(figs/f'P04_{c}_PSD_after_PSD_channel_rejection.png'),
                          f'{c}: PSD after PSD-based channel rejection',
                          f'Welch PSD of remaining good EEG channels after rejecting {fmt_channels(new)} based on PSD inspection.',
                          'All-channel EEG preprocessing')
        plt.close(fig)
        audit['conditions'][c]={'n_good_eeg_channels':len(good),'good_eeg_channels':good,
                                'psd':{'method':'welch','fmin_hz':0.1,
                                       'fmax_hz':min(100.,ep.info['sfreq']/2.),'n_fft':n}}

    audit['changed_clean_files']=True
    (qc_dir(root,s)/'P04_psd_channel_rejection.json').write_text(json.dumps(audit,indent=2)+'\n')
    report.add_text('Post-PSD channel rejection',
                    f'Additional channels rejected after PSD inspection: {fmt_channels(new)}. Reasons: '+
                    '; '.join(f'{ch}: {reason}' for ch,reason in reasons.items())+
                    f'. Final bad channels: {fmt_channels(final)}. Clean epoch files were updated for both conditions and PSD was replotted.',
                    'All-channel EEG preprocessing')
    print(f'PSD-based channel rejection complete. Updated PDF: {report.pdf_fname}')


if __name__=='__main__': main()
