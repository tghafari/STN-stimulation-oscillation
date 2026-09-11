"""Semi-automatic all-channel EEG runner: preprocessing -> ERP -> TFR -> report summary.

The repository is organised into preprocessing/ and sensor/ subdirectories. This
runner resolves those paths explicitly and runs the complete participant workflow
while preserving all manual QC pauses inside the individual scripts.
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Allow the runner and child scripts to import the shared all-channel modules even
# though the executable scripts themselves now live in subdirectories.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from pipeline_config import qc_dir, resolve_project_root

PREPROCESSING = HERE / 'preprocessing'
SENSOR = HERE / 'sensor'
STAGES = [
    (0, 'posterior-alpha inclusion QC', PREPROCESSING / 'P00_screen_subject.py'),
    (1, 'full continuous LP100 + PyPREP -> stim segmentation -> epochs -> rereference', PREPROCESSING / 'P01_epoch_pyprep_reref.py'),
    (2, 'ICA fit on concatenated rereferenced stim/no-stim epochs and applied separately', PREPROCESSING / 'P02_ica.py'),
    (3, 'manual rejection of ICA-cleaned stim/no-stim epochs + final spectral QC', PREPROCESSING / 'P03_manual_epoch_rejection.py'),
    (4, 'ERP: scalp-layout sensors, PO3/POz/PO4, posterior mean', SENSOR / 'A01_ERP.py'),
    (5, 'TFR: condition maps, stim-no-stim difference and normalized difference', SENSOR / 'A02_TFR.py'),
    (6, 'manuscript-style participant analysis overview', SENSOR / 'A03_analysis_overview.py'),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--subjects', nargs='+')
    g.add_argument('--range', nargs=2, type=int)
    p.add_argument('--session', default='01')
    p.add_argument('--task', default='SpAtt')
    p.add_argument('--run', default='01')
    p.add_argument('--platform', choices=['mac', 'bluebear'], default='mac')
    p.add_argument('--project-root', default=None)
    p.add_argument('--from-stage', type=int, choices=range(7), default=0)
    p.add_argument('--continue-on-error', action='store_true')
    return p.parse_args()


def subjects(a):
    if a.subjects:
        return [str(x).removeprefix('sub-') for x in a.subjects]
    return [str(x) for x in range(a.range[0], a.range[1] + 1)]


def status(root, subject):
    path = qc_dir(root, subject) / 'inclusion_decision.json'
    return json.loads(path.read_text()).get('decision') if path.exists() else None


def run_stage(script, subject, a):
    if not script.exists():
        raise FileNotFoundError(f'Pipeline stage script not found: {script}')

    cmd = [
        sys.executable,
        str(script),
        '--subject', subject,
        '--platform', a.platform,
    ]
    if script.name != 'A03_analysis_overview.py':
        cmd += ['--session', a.session, '--task', a.task, '--run', a.run]
    if a.project_root:
        cmd += ['--project-root', a.project_root]

    # Child scripts are in subfolders but import pipeline_config and
    # all_channel_report from HERE. Explicit PYTHONPATH makes this robust whether
    # the runner is launched from IPython, VS Code, or a terminal.
    env = dict(__import__('os').environ)
    existing = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = str(HERE) + (__import__('os').pathsep + existing if existing else '')

    print('\nRUNNING: ' + ' '.join(cmd))
    subprocess.run(cmd, check=True, env=env)


def main():
    a = parse_args()
    root = resolve_project_root(a.platform, a.project_root)

    print('\nAll-channel EEG pipeline')
    print('Pipeline root:', HERE)
    print('Preprocessing scripts:', PREPROCESSING)
    print('Sensor-analysis scripts:', SENSOR)

    for subject in subjects(a):
        print('\n' + '=' * 78)
        print(f'STARTING sub-{subject}')
        print('=' * 78)
        try:
            for number, label, script in STAGES:
                if number < a.from_stage:
                    continue
                if number > 0 and status(root, subject) != 'include':
                    print(f'Stopping sub-{subject}: inclusion status is {status(root, subject)!r}.')
                    break
                print(f'\nStage {number}: {label}')
                run_stage(script, subject, a)
        except Exception as exc:
            print(f'ERROR sub-{subject}: {type(exc).__name__}: {exc}')
            if not a.continue_on_error:
                raise


if __name__ == '__main__':
    main()
