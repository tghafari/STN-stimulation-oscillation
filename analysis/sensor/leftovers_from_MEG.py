ROI_fname = op.join(ROI_dir, f'sub-{subject}_ROI.csv')
MI_ALI_fname = op.join(ROI_dir, f'sub-{subject}_MI_ALI.csv')
ROI_MI_ALI_fname = op.join(ROI_dir, f'sub-{subject}_ROI_MI_ALI.csv')
ROI_MI_ALI_html =  op.join(ROI_dir, f'sub-{subject}_ROI_MI_ALI.html')

# Read sensor layout sheet from camcan RDS
"""these variables are in correct right-and-left-corresponding-sensors order"""
sensors_layout_sheet = op.join(project_root, 'results/sensor-layout-with-centre.xlsx')
sensors_layout_names_dict = pd.read_excel(sensors_layout_sheet, sheet_name=None)
sensors_layout_names_df = sensors_layout_names_dict['Sheet1']

right_sensors = [ch for ch in sensors_layout_names_df['right_sensors']]
left_sensors = [ch for ch in sensors_layout_names_df['left_sensors']]
sensors_layout_df = pd.DataFrame({'left_sensors': left_sensors,
                                  'right_sensors': right_sensors})  

# Sort the right MI DataFrame by MI value and extract the first 5 channel names and save the ROI sensors
df_sorted = MI_right_sens_df.sort_values(by='MI_right', ascending=False, key=abs)
MI_right_ROI = df_sorted.head(5) # create a df of MI right ROI sensors and their MI values
MI_right_ROI = MI_right_ROI.sort_index()  # to ensure the order or sensor names is correct in right and left
ROI_right_sens = MI_right_ROI['right_sensors'].tolist() 

# Find the corresponding left sensors 
ROI_symmetric = sensors_layout_df[sensors_layout_df['right_sensors'].isin(ROI_right_sens)]  # reorders channels by channem name
ROI_symmetric.to_csv(ROI_fname, index=False)

ROI_left_sens = ROI_symmetric['left_sensors'].to_list()

# Calculate MI for right ROI for later plotting
tfr_right_post_stim_alpha_right_ROI_sens = tfr_right_alpha_all_sens.copy().pick(ROI_right_sens).crop(tmin=0.2, tmax=.8)
tfr_left_post_stim_alpha_right_ROI_sens = tfr_left_alpha_all_sens.copy().pick(ROI_right_sens).crop(tmin=0.2, tmax=.8)

tfr_alpha_MI_right_ROI = tfr_right_post_stim_alpha_right_ROI_sens.copy()
tfr_alpha_MI_right_ROI.data = (tfr_right_post_stim_alpha_right_ROI_sens.data - tfr_left_post_stim_alpha_right_ROI_sens.data) \
    / (tfr_right_post_stim_alpha_right_ROI_sens.data + tfr_right_post_stim_alpha_right_ROI_sens.data)  # shape: #sensors, #freqs, #time points

# ========================================= LEFT SENSORS and ROI ON TOPOMAP (FIFTH PLOT) =======================================
# Crop tfrs to post-stim alpha and right sensors
tfr_right_post_stim_alpha_left_ROI_sens = tfr_right_alpha_all_sens.copy().pick(ROI_left_sens).crop(tmin=0.2, tmax=.8)
tfr_left_post_stim_alpha_left_ROI_sens = tfr_left_alpha_all_sens.copy().pick(ROI_left_sens).crop(tmin=0.2, tmax=.8)

# Calculate power modulation for attention right and left (always R- L)
tfr_alpha_MI_left_ROI = tfr_left_post_stim_alpha_left_ROI_sens.copy()
tfr_alpha_MI_left_ROI.data = (tfr_right_post_stim_alpha_left_ROI_sens.data - tfr_left_post_stim_alpha_left_ROI_sens.data) \
    / (tfr_right_post_stim_alpha_left_ROI_sens.data + tfr_left_post_stim_alpha_left_ROI_sens.data)  # shape: #sensors, #freqs, #time points

# Average across time points and alpha frequencies
tfr_avg_alpha_MI_left_ROI_power = np.mean(tfr_alpha_MI_left_ROI.data, axis=(1,2))   # the order of channels is the same as right_sensors (I double checked)

# Save to dataframe
MI_left_ROI_df = pd.DataFrame({'MI_left': tfr_avg_alpha_MI_left_ROI_power,
                               'left_sensors': ROI_left_sens})  

# ========================================= MI OVER TIME AND SIXTH PLOT =======================================
# Plot MI avg across ROI over time
fig, axs = plt.subplots(figsize=(12, 6))

# Plot average power and std for tfr_alpha_MI_left_ROI
axs.plot(tfr_alpha_MI_occ_chans.times, tfr_alpha_MI_occ_chans.data.mean(axis=(0, 1)), label='Average MI', color='red')
axs.fill_between(tfr_alpha_MI_occ_chans.times,
                    tfr_alpha_MI_occ_chans.data.mean(axis=(0, 1)) - tfr_alpha_MI_occ_chans.data.std(axis=(0, 1)),
                    tfr_alpha_MI_occ_chans.data.mean(axis=(0, 1)) + tfr_alpha_MI_occ_chans.data.std(axis=(0, 1)),
                    color='red', alpha=0.3, label='Standard Deviation')
axs.set_title('MI on occipital and parietla channels')
axs.set_xlabel('Time (s)')
axs.set_ylabel('Average MI (PAF)')
axs.set_ylim(min(tfr_alpha_MI_occ_chans.data.mean(axis=(0, 1))) - 0.3, 
                    max(tfr_alpha_MI_occ_chans.data.mean(axis=(0, 1))) + 0.3)
axs.legend()

# Plot average power and std for tfr_alpha_MI_right_ROI
#axs[1].plot(tfr_alpha_MI_right_ROI.times, tfr_alpha_MI_right_ROI.data.mean(axis=(0, 1)), label='Average MI', color='blue')
#axs[1].fill_between(tfr_alpha_MI_right_ROI.times,
#                    tfr_alpha_MI_right_ROI.data.mean(axis=(0, 1)) - tfr_alpha_MI_right_ROI.data.std(axis=(0, 1)),
#                    tfr_alpha_MI_right_ROI.data.mean(axis=(0, 1)) + tfr_alpha_MI_right_ROI.data.std(axis=(0, 1)),
#                    color='blue', alpha=0.3, label='Standard Deviation')
#axs[1].set_title('MI on right ROI')
#axs[1].set_xlabel('Time (s)')
#axs[1].set_ylabel('Average MI (PAF)')
#axs[0].set_ylim(min(tfr_alpha_MI_right_ROI.data.mean(axis=(0, 1))) - 0.3, 
#                   max(tfr_alpha_MI_right_ROI.data.mean(axis=(0, 1))) + 0.3)
#axs[1].legend()

# Adjust layout
plt.tight_layout()

# Save the figure in a variable
fig_mi_overtime = fig

# Show plot (optional)
plt.show()

# ========================================= ALI AND PRIMARY OUTCOME (LAST OUTPUT) =======================================
ALI = np.mean(MI_right_ROI['MI_right']) - np.mean(MI_left_ROI_df['MI_left'])
ROI_ALI_df = pd.DataFrame({'ALI_avg_ROI':[ALI]})  # scalars should be lists for dataframe conversion

# Save and read the dataframe as html for the report
ROI_ALI_df.to_html(ROI_MI_ALI_html)
with open(ROI_MI_ALI_html, 'r') as f:
    html_string = f.read()
