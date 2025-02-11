"""
===============================================
00_testing_LFP_data
    this code opens the csv output from the LFP
    device and plots the microvolte value of 
    each sample and the tag code that was sent 
    as trigger.
    the output is a csv file with three columns,
    one: sample number
    two: microvolte value for each sample
    three: is the tag code that is 
    used as trigger.

    this does not input the edf.
    for edf use 01_first_look
        
    
written by Tara Ghafari
t.ghafari@bham.ac.uk
==============================================  

ToDos:
    check the sfreq for different subjects 
        it is 1250Hz


"""

# Import relevant Python modules
import os.path as op
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


# fill these out
subject = '103'  # subject code assigned to by Benchi's group
session = '01'  # name of the folder containing the lfp data
modality = 'lfp'
side = 'lfp left'  # 'lfp right'
csv_fname = '1010P24295_2025_01_15_09_57_47_uv.csv'
platform = 'mac'  # are you using 'bluebear', 'mac', or 'windows'?
pilot = False  # is it pilot data or real data?
rprt = True

sfreq = 1000  #the correct sfreq is 1250  

if platform == 'bluebear':
    rds_dir = '/rds/projects/j/jenseno-avtemporal-attention-1'
elif platform == 'mac':
    rds_dir = '/Volumes/jenseno-avtemporal-attention'

project_root = op.join(rds_dir, 'Projects/subcortical-structures/STN-in-PD')
# '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/STN-in-PD'  # only for bear outage time
data_root = op.join(project_root, 'data/data-organised')
# '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/STN-in-PD/data/data-organised'  # only for bear outage time

base_fpath = op.join(data_root, f'sub-{subject}', f'ses-{session}', f'{modality}', f'{side}')  
csv_lfp_fname = op.join(base_fpath, csv_fname)

# Reading the lfp csv file as a dataframe
lfp_df = pd.read_csv(csv_lfp_fname, on_bad_lines='skip')
lfp_df = lfp_df.rename(columns={'Device Type':'sample', 'WCH':'amplitude', 'Unnamed: 2':'tag_code'})

# Convert columns to numeric, coercing errors, then drop rows with NaNs
lfp_df.iloc[:, 0] = pd.to_numeric(lfp_df.iloc[:, 0], errors='coerce')
lfp_df.iloc[:, 1] = pd.to_numeric(lfp_df.iloc[:, 1], errors='coerce')

# Exclude the first three rows that containt text information
lfp_df_excluded = lfp_df.iloc[3:]

# Line plot: voltage per sample
plt.figure(figsize=(10, 6))
plt.plot(lfp_df_excluded['sample'], lfp_df_excluded['amplitude'], label='raw data')
plt.xlabel(f'samples (sfreq={sfreq})')
plt.ylabel('amplitude (microvolts)')
plt.title(f'Raw Data sub_{subj_code}-{session_code}')

plt.legend()
plt.grid(True)
plt.show()

# Remove empty cells from the tag column
lfp_df['tag_code'].replace('  ', np.nan, inplace=True)
lfp_df_dropna = lfp_df_excluded.dropna()  

# Histogram: number of tag codes that were sent 
plt.figure(figsize=(10, 6))
n, bins, patches = plt.hist(lfp_df_dropna['tag_code'], bins=50, edgecolor='black')

plt.xlabel('tag code')
plt.ylabel('Frequency')
plt.title('Histogram of Tag Codes')
plt.grid(True)

# Write the actual frequency of each value on top of the bar
for i in range(len(patches)):
    plt.text(patches[i].get_x() + patches[i].get_width() / 2, patches[i].get_height(), 
             str(int(patches[i].get_height())), ha='center', va='bottom')
plt.show()