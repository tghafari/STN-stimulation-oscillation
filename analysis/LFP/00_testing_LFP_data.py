"""
===============================================
00_testing_LFP_data
    this code opens the output from the LFP
    device and plots the microvolte value of 
    each sample and the tag code that was sent 
    as trigger.
    the output is a csv file with three columns,
    one: sample number
    two: microvolte value for each sample
    three: is the tag code that is 
    used as trigger.
        
    
written by Tara Ghafari
t.ghafari@bham.ac.uk
==============================================  

ToDos:
    check the sfreq for different subjects


"""

# Import relevant Python modules
import os.path as op
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


# fill these out
subj_code = '01_ly'  # subject code assigned to by Benchi's group
session_code = '01_ly_LFP'  # name of the folder containing the lfp data
csv_fname = '01_ly_left_LFP_blocks1-2.csv'
platform = 'mac'  # are you using 'bluebear', 'mac', or 'windows'?
pilot = True  # is it pilot data or real data?
rprt = True

sfreq = 1000  # the sampling frequency of the lfp data (1000 for ly and wmf 2000 for others) !!TODO:is this correct?

if platform == 'bluebear':
    rds_dir = '/rds/projects/j/jenseno-avtemporal-attention'
elif platform == 'mac':
    rds_dir = '/Volumes/jenseno-avtemporal-attention'

project_root = op.join(rds_dir, 'Projects/subcortical-structures/STN-in-PD')
if pilot:
    data_root = op.join(project_root, 'Data/pilot-data')
else:
    data_root = op.join(project_root, 'Data/real-data')

csv_lfp_fname = op.join(data_root, subj_code, session_code, csv_fname)

# Reading the lfp csv file as a dataframe
lfp_df = pd.read_csv(csv_lfp_fname, on_bad_lines='skip')
lfp_df = lfp_df.rename(columns={'Device Type':'sample', 'WCH':'amplitude', 'Unnamed: 2':'tag_code'})

# Convert columns to numeric, coercing errors, then drop rows with NaNs
lfp_df.iloc[:, 0] = pd.to_numeric(lfp_df.iloc[:, 0], errors='coerce')
lfp_df.iloc[:, 1] = pd.to_numeric(lfp_df.iloc[:, 1], errors='coerce')

# Remove empty cells from the tag column
lfp_df['tag_code'].replace('', np.nan, inplace=True)

# Exclude the first three rows
lfp_df_excluded = lfp_df.iloc[3:]
lfp_df_excluded = lfp_df_excluded.dropna()  

# Line plot: voltage per sample
plt.figure(figsize=(10, 6))
plt.plot(lfp_df_excluded.iloc[:, 0], lfp_df_excluded.iloc[:, 1], label='raw data')
plt.xlabel('time (sec)')
plt.ylabel('amplitude (microvolts)')
plt.title(f'Raw Data sub_{subj_code}')
plt.legend()
plt.grid(True)
plt.show()

# Histogram: number of tag codes that were sent 
plt.figure(figsize=(10, 6))
plt.hist(lfp_df_excluded.iloc[:, 2], bins=50, edgecolor='black')
plt.xlabel('tag code')
plt.ylabel('Frequency')
plt.title('Histogram of Tag Codes')
plt.grid(True)
plt.show()