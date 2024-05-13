"""
===============================================
02_reading_beh_data
    this code opens the .mat file from MATLAB
    (output of psychtoolbox) and computes RT
    !! not complete !!

written by Tara Ghafari
t.ghafari@bham.ac.uk
==============================================  

ToDos:
- read the .mat file into csv OR add save into csv
to the psychtoolbox code


"""

# Import relevant Python modules
import os.path as op
import os
import pandas as pd
import scipy.io

import matplotlib.pyplot as plt
from mne_bids import BIDSPath

# fill these out
subj_code = '01_ly'  # subject code assigned to by Benchi's group
subject_id = 'S101'
platform = 'mac'  # are you using 'bluebear', 'mac', or 'windows'?
pilot = True  # is it pilot data or real data?
rprt = True

if platform == 'bluebear':
    rds_dir = '/rds/projects/j/jenseno-avtemporal-attention'
elif platform == 'mac':
    rds_dir = '/Volumes/jenseno-avtemporal-attention'

project_root = op.join(rds_dir, 'Projects/subcortical-structures/STN-in-PD')
if pilot:
    data_root = op.join(project_root, 'Data/pilot-data/AO')
else:
    data_root = op.join(project_root, 'Data/real-data')

# BIDS settings
bids_root = op.join(project_root, 'Data', 'BIDS')
subject = '01'
session = '01'
task = 'SpAtt'
run = '01'
bids_path = BIDSPath(subject=subject, session=session, datatype ='beh',
                     task=task, run=run, root=bids_root)

base_fname = op.join(data_root, subj_code, f'{subj_code}_behavioral', f'ses-{session}', 'eeg')
mat_beh_fname = op.join(base_fname, f'sub-{subject_id}_ses-{session}_task-{task.lower()}_run-{run}_logfile' + '.mat')
csv_beh_fname = str(bids_path.fpath) + '_beh.csv'

# Read the behavioural data of .mat and save as csv  - this doesn't work
behaviour_mat = scipy.io.loadmat(mat_beh_fname)
flat_data = {key: value.flatten() for key, value in behaviour_mat.items()}
data = pd.DataFrame(flat_data) 

data.to_csv("example.csv")