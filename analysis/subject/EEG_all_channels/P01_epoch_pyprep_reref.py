"""P01: full-recording PyPREP -> PSD/manual channel QC -> stim segmentation -> epochs -> rereference.

Exact order
-----------
1. Read original unsegmented continuous BIDS EEG and low-pass it at 100 Hz.
2. Run PyPREP once on the full continuous recording.
3. Mark PyPREP suggestions and SHOW the continuous-data PSD interactively.
4. Researcher adds bad channels or rescues PyPREP channels judged to be good.
5. Segment stim/no-stim using stimulation_cropped_time.json.
6. Define cue epochs (-0.5 to +1.6 s) separately in each segment.
7. Average-rereference epochs using good EEG channels; bads remain marked.
"""
from __future__ import annotations
import argparse,json
import matplotlib.pyplot as plt
import mne
from mne_bids import read_raw_bids
from pyprep.find_noisy_channels import NoisyChannels
from pipeline_config import CONDITIONS,base_bids_path,crop_table_path,qc_dir,resolve_project_root,stage_path
from all_channel_report import participant_report,figure_dir,fmt_channels
EVENT_DICT={'cue_onset_right':1,'cue_onset_left':2,'trial_onset':3,'stim_onset':4,'catch_onset':5,'dot_onset_right':6,'dot_onset_left':7,'response_press_onset':8,'block_onset':20,'block_end':21,'experiment_end':30,'new_stim_segment':99999}
def parse_args():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--subject',required=True); p.add_argument('--session',default='01'); p.add_argument('--task',default='SpAtt'); p.add_argument('--run',default='01'); p.add_argument('--platform',choices=['mac','bluebear'],default='mac'); p.add_argument('--project-root',default=None); return p.parse_args()
def pyprep_reasons(raw):
    eeg=raw.copy().pick('eeg')
    if eeg.get_montage() is None:eeg.set_montage('standard_1020',on_missing='warn')
    noisy=NoisyChannels(eeg,random_state=42); errors=[]
    for label,func in [('deviation',noisy.find_bad_by_deviation),('high-frequency noise',noisy.find_bad_by_hfnoise),('correlation',noisy.find_bad_by_correlation),('RANSAC',noisy.find_bad_by_ransac)]:
        try:func()
        except Exception as exc:errors.append(f'{label}: {type(exc).__name__}: {exc}')
    amap={'bad_by_deviation':'deviation','bad_by_hf_noise':'high-frequency noise','bad_by_correlation':'correlation','bad_by_ransac':'RANSAC','bad_by_nan':'NaN/flat data','bad_by_SNR':'poor signal-to-noise ratio'}; reasons={}
    for attr,why in amap.items():
        for ch in getattr(noisy,attr,[]) or []:reasons.setdefault(str(ch),[]).append(why)
    for ch in noisy.get_bads():reasons.setdefault(str(ch),[]).append('PyPREP overall decision')
    return {k:sorted(set(v)) for k,v in reasons.items()},errors
def make_segment(raw,times):
    if len(times) not in (2,4):raise ValueError(f'Crop times must contain 2 or 4 values, got {times}')
    pieces=[raw.copy().crop(tmin=float(times[0]),tmax=float(times[1]))]
    if len(times)==4:pieces.append(raw.copy().crop(tmin=float(times[2]),tmax=float(times[3])))
    return pieces[0] if len(pieces)==1 else mne.concatenate_raws(pieces,on_mismatch='warn')
def define_epochs(raw):
    events,event_ids=mne.events_from_annotations(raw,event_id=EVENT_DICT); cue={k:event_ids[k] for k in ('cue_onset_right','cue_onset_left') if k in event_ids}
    if not cue:raise RuntimeError('No cue onset events found after stimulation segmentation.')
    return mne.Epochs(raw,events,cue,tmin=-0.5,tmax=1.6,baseline=None,detrend=1,proj=True,picks='all',reject=None,reject_by_annotation=False,preload=True,event_repeated='merge')
def main():
    a=parse_args(); s=a.subject.removeprefix('sub-'); root=resolve_project_root(a.platform,a.project_root); report=participant_report(root,s); figs=figure_dir(root,s)
    raw=read_raw_bids(base_bids_path(root,s,a.session,a.task,a.run),verbose=True,extra_params={'preload':True})
    if raw.get_montage() is None:raw.set_montage('standard_1020',on_missing='warn')
    raw.filter(l_freq=None,h_freq=100.0)

    reasons,errors=pyprep_reasons(raw); suggested=sorted(reasons)
    raw.info['bads']=sorted(set(raw.info['bads'])|set(suggested))
    print('\nPyPREP suggested bad channels:',suggested or 'None')
    if reasons:
        print('PyPREP reasons:')
        for ch,why in sorted(reasons.items()):print(f'  {ch}: {", ".join(why)}')
    if errors:
        print('PyPREP detector errors:')
        for err in errors:print('  '+err)

    # One PSD object is used both for the persistent PDF and the interactive QC.
    # PyPREP suggestions are already in raw.info["bads"], so MNE marks them in
    # the PSD display. The interactive figure MUST be inspected before answering.
    spectrum=raw.compute_psd(fmin=0.5,fmax=min(100.0,raw.info['sfreq']/2.0))
    fig_report=spectrum.plot(show=False)
    report.add_figure(fig_report,str(figs/'P01_full_continuous_PyPREP_PSD.png'),'Full continuous EEG: PyPREP channel QC','Whole unsegmented recording after 100-Hz low-pass. PyPREP suggestions were marked bad before this PSD was generated.','All-channel preprocessing')
    plt.close(fig_report)
    print('\nOpening PSD for manual bad-channel QC.')
    print('Inspect the PyPREP-marked channels and all other EEG channels, then close the PSD window.')
    spectrum.plot(show=True,block=True)

    additions=input('Additional bad EEG channels (space-separated, Enter for none): ').strip().split()
    removals=input('PyPREP channels you believe are GOOD (space-separated, Enter for none): ').strip().split()
    unknown_add=[ch for ch in additions if ch not in raw.ch_names]; unknown_remove=[ch for ch in removals if ch not in raw.ch_names]
    if unknown_add or unknown_remove:raise ValueError(f'Unknown channel name(s): additions={unknown_add}, removals={unknown_remove}')
    final=sorted(((set(raw.info['bads'])|set(additions)|set(suggested))-set(removals))&set(raw.ch_names)); raw.info['bads']=final
    print('Final bad channels after manual QC:',final or 'None')

    table=json.loads(crop_table_path().read_text(encoding='utf-8')); key=f'sub-{s}'
    if key not in table:raise KeyError(f'No stimulation crop times for {key} in {crop_table_path()}')
    audit={'subject':key,'low_pass_hz':100,'pyprep_reasons':reasons,'detector_errors':errors,'manual_additions':additions,'manual_removals':removals,'bad_channels':final,'conditions':{}}
    for condition in CONDITIONS:
        times=table[key][condition]; segment=make_segment(raw,times); segment.info['bads']=[ch for ch in final if ch in segment.ch_names]
        epochs=define_epochs(segment); epochs.info['bads']=[ch for ch in final if ch in epochs.ch_names]
        epochs.set_eeg_reference(ref_channels='average',projection=False)
        out=stage_path(root,s,a.session,a.task,a.run,condition,'reref','epo'); epochs.save(out,overwrite=True)
        audit['conditions'][condition]={'crop_times_sec':times,'n_epochs':len(epochs)}
        report.add_text(f'{condition}: stimulation segmentation',f'Kept ranges (s): {times}\nWhole recording was low-pass filtered at 100 Hz before segmentation.\nCue epochs: -0.5 to +1.6 s; baseline=None; detrend=1.\nEpochs: {len(epochs)}','Stimulation segmentation')
    reason_text='\n'.join(f'{ch}: {", ".join(v)}' for ch,v in sorted(reasons.items())) or 'None'
    report.add_text('PyPREP and rereferencing',f'PyPREP was run once on the full unsegmented continuous EEG after 100-Hz low-pass.\nThe PSD was inspected manually after PyPREP suggestions were marked.\nDetectors: deviation, high-frequency noise, correlation, RANSAC.\nSuggestions and reasons:\n{reason_text}\nDetector errors: {errors or "None"}\nManual additions: {fmt_channels(additions)}\nManual removals/rescued PyPREP channels: {fmt_channels(removals)}\nFinal bad channels: {fmt_channels(final)}\nAfter stim/no-stim segmentation and cue epoching, each condition was average-rereferenced using good EEG channels. Bad channels remained marked and were not interpolated or dropped.','All-channel preprocessing')
    (qc_dir(root,s)/'P01_pyprep_segment_epoch_reref.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()
