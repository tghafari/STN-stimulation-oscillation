function [cfgStim, cfgExp, cfgTrigger] = read_visual_stim(cfgFile, cfgExp, cfgStim, cfgTrigger,randidx)
% [cfgStim, cfgExp, cfgTrigger] = read_visual_stim(cfgFile, cfgExp, cfgStim, cfgTrigger)
% randomly reads the visual stimuli
% inputs are the directory of stimuli images and number of trials/stim

fileDirStim = dir([cfgFile.stim, '*.bmp']);  % only use the files ending in .bmp (not unwanted files)
fileDirCue = dir([cfgFile.cue, '*.jpg']);

[~,idx] = sort(str2double(regexp({fileDirStim.name},'(?<=cube3D)\d+','match','once'))); % sort the file names increasingly  
cfgStim.fNameStimSortd = fileDirStim(idx);

cfgStim.visStim = cell(length(1:cfgStim.stimRotSpeed:length(cfgStim.fNameStimSortd)), 1);  % preallocation
cfgStim.cueStim = cell(cfgExp.numStim, 1);  % preallocation

% read stimulus images 
for spd = 1:cfgStim.stimRotSpeed:length(cfgStim.fNameStimSortd)
    cfgStim.visStim{spd} = imread(cfgStim.fNameStimSortd(spd).name);
end
cfgStim.visStim = cfgStim.visStim(~cellfun('isempty', cfgStim.visStim'));  % remove indices that are empty due to reading images based on speed

% Ensure equal number of right and left cues
rng('shuffle')
right_cue = ones(cfgExp.numTrial/2, 1);  % random index for cue - 1:right, 2:left
left_cue = 2 * ones(cfgExp.numTrial/2, 1); 
both_cues = squeeze(cat(1, right_cue, left_cue)); 

for i = 1:cfgExp.numBlock
    randomised_cue(:,i) = both_cues(cfgExp.randid(:, i), :);  % randomise using the same random idx from initialise_exp_vars
end

cfgStim.cueRndIdx = reshape(randomised_cue, [], 1); 

for stim = 1:cfgExp.numStim
    cfgStim.cueStim{stim,1} = imread(fileDirCue(cfgStim.cueRndIdx(stim)).name);  % read cue randomly
end

% collect correct responses +
% cue, stim and dot triggers
cfgExp.cuesDir = cell(cfgExp.numStim, 1);  % preallocation
cfgTrigger.cuesDir = cell(cfgExp.numStim, 2);  % preallocation
cfgTrigger.dotDir = cell(cfgExp.numStim, 2);  % preallocation

cfgExp.cuesDir(find(cfgStim.cueRndIdx == 1), 1) = {'Right'};
cfgExp.cuesDir(find(cfgStim.cueRndIdx == 2), 1) = {'Left'};
cfgExp.cuesDir(find(cfgExp.corrResp == 0), 1) = {'no resp'};
cfgTrigger.cuesDir(find(cfgStim.cueRndIdx == 1), 1) = {'1'};  % EEG trigger codes are 1 -> cue right, 2 -> cue left
cfgTrigger.cuesDir(find(cfgStim.cueRndIdx == 2), 1) = {'2'};  % EEG trigger codes are 1 -> cue right, 2 -> cue left
cfgTrigger.cuesDir(find(cfgStim.cueRndIdx == 1), 2) = {'cue_right'};  % trigger message for Eyelink
cfgTrigger.cuesDir(find(cfgStim.cueRndIdx == 2), 2) = {'cue_left'};  
cfgTrigger.dotDir(find(cfgStim.cueRndIdx == 1), 1) = {'6'};  % EEG trigger codes are 6 -> dot right, 7 -> dot left
cfgTrigger.dotDir(find(cfgStim.cueRndIdx == 2), 1) = {'7'};  % EEG trigger codes are 6 -> dot right, 7 -> dot left
cfgTrigger.dotDir(find(cfgStim.cueRndIdx == 1), 2) = {'dot_right'};  % trigger message for Eyelink
cfgTrigger.dotDir(find(cfgStim.cueRndIdx == 2), 2) = {'dot_left'};  
cfgTrigger.dotDir(find(cfgExp.corrResp == 0), [1, 2]) = {'no resp'};

end