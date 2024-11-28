function cfgFile = create_file_directory(cfgExp)
%[fileDirStim,fileDirRes,fileDirCue] = create_file_directory(cfgExp)
%   cd to and creates subject directory according to OS

tmp = matlab.desktop.editor.getActive;
cd(fileparts(tmp.Filename));  % move to the current directory
cd ..  % move up one folder

addpath([cd, filesep 'Experiment_Functions' filesep]);  % add the sub-functions
addpath([cd, filesep 'Eyelink' filesep]);  % add Eyelink functions
addpath([cd, filesep 'Stimuli' filesep]);  % add stimulus images
addpath([cd, filesep 'Results' filesep]);  % add result folder

if strcmp(cfgExp.answer.pc,'win')
    cfgFile.res = [cd, filesep 'Results' filesep] ; 
    cfgFile.stim = [cd, filesep 'Stimuli' filesep 'Visual_Stimuli' filesep];
    cfgFile.cue = [cd, filesep 'Stimuli' filesep 'Cue_Stimuli' filesep];

elseif strcmp(cfgExp.answer.pc,'EEG')
    cfgFile.res = [cd, filesep 'Results' filesep] ; 
    cfgFile.stim = [cd, filesep 'Stimuli' filesep 'Visual_Stimuli' filesep];
    cfgFile.cue = [cd, filesep 'Stimuli' filesep 'Cue_Stimuli' filesep];

end

mkdir([cfgFile.res, 'sub-', cfgExp.answer.sub, filesep, 'ses-', cfgExp.answer.ses, filesep, 'eeg', filesep]);  % make result directory with BIDS format
cfgFile.subDir = [cfgFile.res, 'sub-', cfgExp.answer.sub, filesep, 'ses-' cfgExp.answer.ses, filesep, 'eeg', filesep];  % store subject directory address
if strcmp(cfgExp.answer.test,'train')
    cfgFile.BIDSname = ['sub-', cfgExp.answer.sub, '_', 'ses-', cfgExp.answer.ses, '_'...
     , 'train-', cfgExp.answer.task, '_', 'run-', cfgExp.answer.run];  % BIDS specific file name
else
cfgFile.BIDSname = ['sub-', cfgExp.answer.sub, '_', 'ses-', cfgExp.answer.ses, '_'...
    , 'task-', cfgExp.answer.task, '_', 'run-', cfgExp.answer.run];  % BIDS specific file name
end
cfgFile.edfFile = ['_eyetracking', '.edf'];  % eyetracking file name
cfgFile.logFile = ['_logfile', '.mat'];  % logfile file name
cfgFile.eyelink = ['e', cfgExp.answer.run, cfgExp.answer.sub];  % file name to use on eyelink pc

cd(fileparts(tmp.Filename));  % move back to the experiment_functions directory for io64.mexw64 file

end

