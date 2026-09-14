"""A03: add a manuscript-style summary derived from the participant audit files.

This stage deliberately reads the audit JSONs written by P01/P02/P03/A01/A02 so
that subject-specific choices (reference, ICA exclusions, final bad channels and
ROI channels) in the report reflect what was actually run rather than stale text.
"""
from __future__ import annotations
import argparse,json
from pipeline_config import qc_dir,resolve_project_root
from all_channel_report import participant_report,fmt_channels
def parse_args():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--subject',required=True);p.add_argument('--platform',choices=['mac','bluebear'],default='mac');p.add_argument('--project-root',default=None);return p.parse_args()
def read_json(path):return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
def main():
 a=parse_args();s=a.subject.removeprefix('sub-');root=resolve_project_root(a.platform,a.project_root);q=qc_dir(root,s);report=participant_report(root,s)
 p01=read_json(q/'P01_pyprep_segment_epoch_reref.json');p02=read_json(q/'P02_ica_decisions.json');p03=read_json(q/'P03_manual_epoch_rejection.json');erp=read_json(q/'A01_erp_analysis.json');tfr=read_json(q/'A02_tfr_analysis.json')
 ref='average EEG reference' if p01.get('rereference')=='avg' else 'original reference (no rereferencing)';final_bad=p03.get('final_bad_channels',p01.get('bad_channels',[]));ica_ex=p02.get('excluded_components',[]);erp_roi=erp.get('roi_final',[]);tfr_roi=tfr.get('roi_final',[])
 text=(
 f'Continuous EEG was band-pass filtered from {p01.get("high_pass_hz",1):g} to {p01.get("low_pass_hz",100):g} Hz before channel-quality assessment. PyPREP was applied to the full continuous recording, followed by manual inspection of the continuous data, review of rejected-channel spectra, and a final spectral check of retained channels. The final bad-channel set was {fmt_channels(final_bad)}. '
 f'Stimulation and no-stimulation periods were segmented using the predefined stimulation timing table. Cue-locked epochs extended from -0.5 to +1.6 s, with no epoch baseline correction and linear detrending (detrend=1). The EEG was then kept with {ref}. '
 f'One FastICA decomposition was fitted to concatenated stimulation and no-stimulation epochs using an ICA-fitting copy resampled to {p02.get("fit_resample_hz",200)} Hz and filtered {p02.get("fit_filter_hz",[1,40])[0]}-{p02.get("fit_filter_hz",[1,40])[1]} Hz. Up to {p02.get("n_components",30)} components were fitted; component maps and source time series were inspected together, and manually excluded components were {ica_ex or "none"}. The fitted ICA solution was applied separately to the original stimulation and no-stimulation epochs. '
 'ICA-cleaned epochs were then inspected manually for bad trials, with the option to mark additional consistently bad EEG channels; any final channel additions were applied to both conditions and remained marked bad rather than being interpolated or physically dropped. Final epoch spectral QC used Welch power spectra from 0.1 Hz to the lower of 100 Hz or the Nyquist frequency, with n_fft limited to approximately 2 s of data. Stimulation-artifact QC consisted of the stimulation and no-stimulation mean Welch PSDs overlaid on one plot using the same common good EEG channels; it was diagnostic only and did not correct the data. '
 f'Cue-locked ERPs combined attention-left and attention-right trials within stimulation condition, were averaged across retained trials, low-pass filtered at {erp.get("evoked_low_pass_hz",30):g} Hz, baseline corrected from {erp.get("baseline_s",[-.1,0])[0]:g} to {erp.get("baseline_s",[-.1,0])[1]:g} s, and displayed from {erp.get("erp_display_window_s",[-.1,.5])[0]:g} to {erp.get("erp_display_window_s",[-.1,.5])[1]:g} s. All common good sensors were displayed in scalp layout and the eight predefined posterior/occipital candidates O7, O3, PO3, POz, Oz, PO4, O4 and O8 were displayed separately before the ROI decision. Final channels contributing to the ERP ROI mean: {fmt_channels(erp_roi)}. '
 f'Time-frequency power combined attention-left and attention-right trials within condition and was estimated with multitaper convolution from 2 to 30 Hz in 1-Hz steps, n_cycles=frequency/2, time-bandwidth={tfr.get("time_bandwidth",2):g}, FFT enabled, trial averaging enabled and decimation={tfr.get("decim",2)}. Stimulation and no-stimulation TFR displays used percent baseline correction from {tfr.get("condition_baseline_s",[-.3,-.1])[0]:g} to {tfr.get("condition_baseline_s",[-.3,-.1])[1]:g} s. Stimulation-minus-no-stimulation and (stimulation-no stimulation)/(stimulation+no stimulation) were calculated from unbaselined power and received no baseline correction. All common good sensors were displayed in scalp layout and the same eight posterior/occipital candidates were displayed separately before ROI selection. Final channels contributing to the TFR ROI mean: {fmt_channels(tfr_roi)}.'
 )
 report.add_text('Participant EEG analysis: methods summary',text,'Analysis overview (manuscript style)');print(f'Analysis overview added for sub-{s}: {report.pdf_fname}')
if __name__=='__main__':main()
