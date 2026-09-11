"""Manual work on segmented/epoched EEG without changing saved data.

Run in IPython, for example:
    %run manual_epoch_exploration.py --subject 103

The script automatically reads whether P01 used no reference or average reference,
loads the corresponding stim/no-stim P01 epoch files, and creates useful PSD/TFR
objects. It does not save or overwrite EEG files.
"""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import mne
import numpy as np
HERE=Path(__file__).resolve().parent; PIPELINE=HERE.parent
if str(PIPELINE) not in sys.path:sys.path.insert(0,str(PIPELINE))
from pipeline_config import CONDITIONS,participant_rereference,reference_desc,resolve_project_root,stage_path
POSTERIOR=['PO3','POz','PO4']
def parse_args():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--subject',required=True);p.add_argument('--session',default='01');p.add_argument('--task',default='SpAtt');p.add_argument('--run',default='01');p.add_argument('--platform',choices=['mac','bluebear'],default='mac');p.add_argument('--project-root',default=None);return p.parse_args()
def main():
 a=parse_args();s=a.subject.removeprefix('sub-');root=resolve_project_root(a.platform,a.project_root);ref=participant_rereference(root,s);print(f'P01 reference state: {reference_desc(ref)}')
 epochs={}
 for c in CONDITIONS:
  path=stage_path(root,s,a.session,a.task,a.run,c,'reref','epo')
  if not path.exists():raise FileNotFoundError(f'Missing P01 epochs: {path}')
  epochs[c]=mne.read_epochs(path,preload=True);print(f'{c}: {path.name} ({len(epochs[c])} epochs)')
 stim=epochs['stim'];nostim=epochs['no-stim']
 # Welch PSD: same settings as final preprocessing QC.
 psd_stim=stim.compute_psd(method='welch',fmin=0.1,fmax=min(100.,stim.info['sfreq']/2.),n_fft=min(int(2*stim.info['sfreq']),len(stim.times)))
 psd_nostim=nostim.compute_psd(method='welch',fmin=0.1,fmax=min(100.,nostim.info['sfreq']/2.),n_fft=min(int(2*nostim.info['sfreq']),len(nostim.times)))
 posterior=[ch for ch in POSTERIOR if ch in stim.ch_names]
 psd_stim_post=stim.copy().pick(posterior).compute_psd(method='welch',fmin=0.1,fmax=min(100.,stim.info['sfreq']/2.),n_fft=min(int(2*stim.info['sfreq']),len(stim.times))) if posterior else None
 # Multitaper TFR: same settings as A02.
 freqs=np.arange(2.,31.,1.);n_cycles=freqs/2.
 tfr_stim=stim.compute_tfr(method='multitaper',freqs=freqs,n_cycles=n_cycles,time_bandwidth=2.,use_fft=True,return_itc=False,average=True,decim=2,n_jobs=4)
 tfr_nostim=nostim.compute_tfr(method='multitaper',freqs=freqs,n_cycles=n_cycles,time_bandwidth=2.,use_fft=True,return_itc=False,average=True,decim=2,n_jobs=4)
 tfr_stim_percent=tfr_stim.copy().apply_baseline((-0.3,-0.1),mode='percent');tfr_nostim_percent=tfr_nostim.copy().apply_baseline((-0.3,-0.1),mode='percent')
 globals().update(locals())
 print('\nObjects ready for manual work: stim, nostim, psd_stim, psd_nostim, psd_stim_post, tfr_stim, tfr_nostim, tfr_stim_percent, tfr_nostim_percent')
 print('\nExamples:')
 print('  stim.plot()')
 print('  psd_stim.plot()')
 print('  psd_nostim.plot()')
 print('  psd_stim_post.plot()')
 print("  tfr_stim_percent.plot(picks='POz', tmin=-0.3, tmax=1.4)")
 print("  tfr_nostim_percent.plot(picks='PO3', tmin=-0.3, tmax=1.4)")
 print('\nUse tfr_stim/tfr_nostim (unbaselined) for difference or ratio calculations.')
if __name__=='__main__':main()
