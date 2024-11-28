function [cfgStim, cfgExp, cfgTrigger] = read_visual_stim(cfgFile, cfgExp, cfgStim, cfgTrigger)

% [cfgStim, cfgExp, cfgTrigger] = read_visual_stim(cfgFile, cfgExp, cfgStim, cfgTrigger)
% Randomly reads and assigns the visual stimuli for the experiment
% inputs are the directory of stimuli images and number of trials/stim

% Get all .bmp files in the stimulus directory, excluding hidden files
fileDirStim = dir([cfgFile.stim, '*.bmp']);
fileDirStim = fileDirStim(arrayfun(@(x) ~strcmp(x.name(1), '.') && ~x.isdir && ~is_hidden_file(x), fileDirStim)); % Exclude hidden files

% Get all .jpg files in the cue directory, excluding hidden files
fileDirCue = dir([cfgFile.cue, '*.jpg']);
fileDirCue = fileDirCue(arrayfun(@(x) ~strcmp(x.name(1), '.') && ~x.isdir && ~is_hidden_file(x), fileDirCue)); % Exclude hidden files

% Sort stimuli by numerical order
[~,idx] = sort(str2double(regexp({fileDirStim.name},'(?<=cube3D)\d+','match','once'))); 
cfgStim.fNameStimSortd = fileDirStim(idx);

% Preallocate for cues and stimuli
cfgStim.visStim = cell(length(1:cfgStim.stimRotSpeed:length(cfgStim.fNameStimSortd)), 1); 
cfgStim.cueStim = cell(cfgExp.numStim, 1); 

% Read stimulus images 
for spd = 1:cfgStim.stimRotSpeed:length(cfgStim.fNameStimSortd)
    cfgStim.visStim{spd} = imread(cfgStim.fNameStimSortd(spd).name);
end
cfgStim.visStim = cfgStim.visStim(~cellfun('isempty', cfgStim.visStim'));  % remove indices that are empty due to reading images based on speed

% Assign cue images based on trial structure
for stim = 1:cfgExp.numStim
    cueIdx = cfgExp.trialMatrix(stim);  % Cue type for this trial (1 = right, 2 = left)
    cfgStim.cueStim{stim} = imread(fileDirCue(cueIdx).name);  % Assign corresponding cue image
end

% Set up triggers for cues and dots
cfgExp.cuesDir = cell(cfgExp.numStim, 1);  % Preallocate for exp info
cfgTrigger.cuesDir = cell(cfgExp.numStim, 2);  % Preallocate for trigger info
cfgTrigger.dotDir = cell(cfgExp.numStim, 2);  

cfgExp.cuesDir(cfgExp.trialMatrix == 1, 1) = {'Right'};
cfgExp.cuesDir(cfgExp.trialMatrix == 2, 1) = {'Left'};
cfgExp.cuesDir(cfgExp.catchMatrix == 1, 1) = {'no_resp'};

cfgTrigger.cuesDir(cfgExp.trialMatrix == 1, :) = {'1', 'cue_right'};  % EEG trigger codes are 1 -> cue right, 2 -> cue left + trigger message for Eyelink
cfgTrigger.cuesDir(cfgExp.trialMatrix == 2, :) = {'2', 'cue_left'};
cfgTrigger.dotDir(cfgExp.trialMatrix == 1, :) = {'6', 'dot_right'};   % EEG trigger codes are 6 -> dot right, 7 -> dot left + trigger message for Eyelink
cfgTrigger.dotDir(cfgExp.trialMatrix == 2, :) = {'7', 'dot_left'};
cfgTrigger.dotDir(cfgExp.catchMatrix == 1, :) = {'no_resp', 'no_resp'};

end