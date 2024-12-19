function [cfgExp, cfgOutput] = initialise_exp_variables(cfgExp)
% [cfgExp, cfgOutput] = initialise_exp_variables(cfgExp)
% Introduces variables of interest for SpAtt task
% to change any repetition you should edit this function

rng('shuffle')
% total time: ~30 minute (5 to 7.5 sec each trial, ~3.6 min each block)

% Parameters
cfgExp.numBlock = 8;  % total number of blocks (8)
cfgExp.numTrial = 40;  % number of trials in each block (40)
cfgExp.numStim = cfgExp.numTrial * cfgExp.numBlock;  % number of stimuli in total
cfgExp.catchPercentage = 0.1;  % percentage of catch trials
cfgExp.numCatchTrialsPerBlock = round(cfgExp.catchPercentage * cfgExp.numTrial);
cfgExp.numRightCatchTrialsPerBlock = cfgExp.numCatchTrialsPerBlock / 2;
cfgExp.numLeftCatchTrialsPerBlock = cfgExp.numCatchTrialsPerBlock / 2;
cfgExp.numRightNonCatchTrialsPerBlock = (cfgExp.numTrial - cfgExp.numCatchTrialsPerBlock) / 2;  % number of non catch trials per side per block
cfgExp.numLeftNonCatchTrialsPerBlock = (cfgExp.numTrial - cfgExp.numCatchTrialsPerBlock) / 2;
cfgExp.numRightCuesPerBlock = cfgExp.numTrial / 2;  % Equal number of right and left cues per block
cfgExp.numLeftCuesPerBlock = cfgExp.numTrial / 2;

% Trial timing parameters
cfgExp.ITIDur =  1000 + (1500 - 1000) .* rand(cfgExp.numStim,1);  % duration of ITI in ms (jitter between 1 and 2 sec)
cfgExp.cueDur = 200;  % duration of cue presentation in ms
cfgExp.ISIDur = 1000;  % interval between cue and grating (stimulus)
cfgExp.stimDur = 1000 + (2000 - 1000) .* rand(cfgExp.numStim,1);  % duration of visual stimulus in ms (jitter between 1 and 3 sec)
cfgExp.dotDur = 50;  % duration of red(white) dot presentation
cfgExp.respTimOut = 3000;  % time during which subject can respond in ms

% Preallocate for output
cfgOutput.presd = zeros(cfgExp.numStim, 1);  % preallocate cfgOutput for unpressed trials
cfgOutput.keyName = cell(cfgExp.numStim, 1);  % preallocate cfgOutput for unpressed trials

% Prompt parameters
if strcmp(cfgExp.answer.pc,'EEG'), cfgExp.EEGLab = 1; else, cfgExp.EEGLab = 0; end  % EEG lab computer-> 1 PC-> 0
if strcmp(cfgExp.answer.test,'task'), cfgExp.task = 1; else, cfgExp.task = 0; end  % are we collecting data and running the task?
if strcmp(cfgExp.answer.test,'train'), cfgExp.train = 1; else, cfgExp.train = 0; end  % are we training the participant?


% Generate balanced and randomized trial structure
trialMatrix = zeros(cfgExp.numTrial, cfgExp.numBlock);  % 1 and 10= right cue, 2 and 20 = left cue

for block = 1:cfgExp.numBlock
    % Generate cues
    cues = [1 * ones(1, cfgExp.numRightNonCatchTrialsPerBlock), 10 * ones(1, cfgExp.numRightCatchTrialsPerBlock),...
            2 * ones(1, cfgExp.numLeftNonCatchTrialsPerBlock), 20 * ones(1, cfgExp.numLeftCatchTrialsPerBlock)];
    cues = cues(randperm(cfgExp.numTrial));  % Randomize cue order
    
    trialMatrix(:, block) = cues;
end

cfgExp.trialMatrix = trialMatrix(:);  % Matrix for right and left cues


end

% generate N random numbers in the interval (a,b): r = a + (b-a).*rand(N,1).
