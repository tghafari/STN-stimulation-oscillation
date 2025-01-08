% check_behavioural_performance

% In this script we load in the behavioural data
% from BIDS structure for each participant.
% Plot RTs that are in cfgOutput
% On the same plot we display performance (%correct).

% Load the behavioural data
subject = '102';
session = '01';
task = 'spatt';
run = '01';
platform = 'mac';  % are you using mac or bluebear

if contains(platform, "mac")
    BIDS_folder = '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage';  % only for bear outage time
    % '/Volumes/jenseno-avtemporal-attention/Projects/subcortical-structures/STN-in-PD/data/BIDS';
else
    BIDS_folder = '/rds/projects/j/jenseno-avtemporal-attention/Projects/subcortical-structures/STN-in-PD/data-organised/BIDS';
end

beh = load([BIDS_folder filesep 'sub-' subject filesep 'ses-' session filesep ...
    'beh' filesep 'sub-S' subject '_ses-' session '_task-' task '_run-' run '_logfile.mat']);

% Extract reaction times
RT_KbQueue = beh.cfgOutput.RT_KbQueue;  % RT from keyboard
RT_trig = beh.cfgOutput.RT_trig;  % RT from triggers

% Calculate preformance 
FB = beh.cfgOutput.presd - beh.cfgExp.corrResp(:);
TPR = sum(FB == 1) ./ length(FB);  % because pressed is stored as 2, TPR = 2 - 1
TNR = sum(FB == 0) ./ length(FB);  % TNR = 0 - 0
FPR = sum(FB == 2) ./ length(FB);  % FPR = 2 - 0
FNR = sum(FB == -1) ./ length(FB);  % FNR = 0 - 1

% Histogram in two subplots RTs with performance measures
figure;
subplot(2,1,1)
histogram(RT_KbQueue, 25);
title('RT from keyboard presses')

subplot(2,1,2)
histogram(RT_trig, 25);
title('RT from triggers')
text(min(xlim)+0.05, max(ylim)-5, sprintf('True Negative Rate = %0.2f%s ', TNR*100, '%'), 'Horiz','left', 'Vert','top')
text(min(xlim)+0.05, max(ylim)-10, sprintf('False Negative Rate = %0.2f%s ', FNR*100, '%'), 'Horiz','left', 'Vert','top')
text(min(xlim)+0.05, max(ylim)-15, sprintf('True Positive Rate = %0.2f%s ', TPR*100, '%'), 'Horiz','left', 'Vert','top')
text(min(xlim)+0.05, max(ylim)-20, sprintf('False Positive Rate = %0.2f%s ', FPR*100, '%'), 'Horiz','left', 'Vert','top')


