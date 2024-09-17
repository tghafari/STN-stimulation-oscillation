function [cfgExp, cfgOutput] = initialise_exp_variables(cfgExp)
% [cfgExp, cfgOutput] = initialise_exp_variables(cfgExp)
% Introduces variables of interest for SpAtt task
% to change any repetition you should edit this function

rng('shuffle')
% total time: ~30 minute (5 to 7.5 sec each trial, ~3.6 min each block)
cfgExp.numBlock = 8;  % total number of blocks (8)
cfgExp.numTrial = 40;  % number of trials in each block (40)
cfgExp.numStim = cfgExp.numTrial * cfgExp.numBlock;  % number of stimuli in total
cfgExp.ITIDur =  1000 + (1500 - 1000) .* rand(cfgExp.numStim,1);  % duration of ITI in ms (jitter between 1 and 2 sec)
cfgExp.cueDur = 200;  % duration of cue presentation in ms
cfgExp.ISIDur = 1000;  % interval between cue and grating (stimulus)
cfgExp.stimDur = 1000 + (2000 - 1000) .* rand(cfgExp.numStim,1);  % duration of visual stimulus in ms (jitter between 1 and 3 sec)
cfgExp.dotDur = 100;  % duration of red dot presentation

% Ensure equal number of right and left and catch in each block 
catch_precentage = 10;  % total percentage of catch trials
catch_trial_per_side = cfgExp.numTrial/2/catch_precentage;
right_resp = ones(cfgExp.numTrial/2 - catch_trial_per_side, 1);
left_resp = ones(cfgExp.numTrial/2 - catch_trial_per_side, 1);
right_catch = zeros(catch_trial_per_side,1); 
left_catch = zeros(catch_trial_per_side,1);
targets_per_block = squeeze(cat(1, right_resp, right_catch, left_resp, left_catch));

% Randomise the order of catch and right and left trials in each block
for i = 1:cfgExp.numBlock  
    randid(:,i) = randperm(length(targets));
    rand_target(:,i) = targets_per_block(randid(:,i));
end
cfgExp.corrResp = rand_target(randid);  % 1=>target present 0=>catch trials
cfgExp.randid = randid;

cfgExp.respTimOut = 3000;  % time during which subject can respond in ms
cfgOutput.presd = zeros(cfgExp.numStim, 1);  % preallocate cfgOutput for unpressed trials
cfgOutput.keyName = cell(cfgExp.numStim, 1);  % preallocate cfgOutput for unpressed trials
  
if strcmp(cfgExp.answer.pc,'EEG'), cfgExp.EEGLab = 1; else, cfgExp.EEGLab = 0; end  % EEG lab computer-> 1 PC-> 0
if strcmp(cfgExp.answer.test,'task'), cfgExp.task = 1; else, cfgExp.task = 0; end  % are we collecting data and running the task?
if strcmp(cfgExp.answer.test,'train'), cfgExp.train = 1; else, cfgExp.train = 0; end  % are we training the participant?

end

% generate N random numbers in the interval (a,b): r = a + (b-a).*rand(N,1).
