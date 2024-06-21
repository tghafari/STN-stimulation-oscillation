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
csv_fname = '01_ly_right_LFP_blocks3-4'
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

csv_lfp_fname = op.join(data_root, subj_code, session_code, csv_fname + '.csv')

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
plt.title(f'Raw Data sub_{subj_code}-{csv_fname}')

plt.legend()
plt.grid(True)
plt.show()

# Remove empty cells from the tag column
lfp_df['tag_code'].replace('  ', np.nan, inplace=True)
lfp_df_dropna = lfp_df_excluded.dropna()  

# Histogram: number of tag codes that were sent 
plt.figure(figsize=(10, 6))
n, bins, patches = plt.hist(lfp_df_dropna['tag_code'], bins=50, edgecolor='black')

#plt.hist(lfp_df_excluded.iloc[:, 2], bins=50, edgecolor='black')
plt.xlabel('tag code')
plt.ylabel('Frequency')
plt.title('Histogram of Tag Codes')
plt.grid(True)

# Write the actual frequency of each value on top of the bar
for i in range(len(patches)):
    plt.text(patches[i].get_x() + patches[i].get_width() / 2, patches[i].get_height(), 
             str(int(patches[i].get_height())), ha='center', va='bottom')
plt.show()