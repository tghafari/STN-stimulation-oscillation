function  cfgTrigger = introduce_triggers
% cfgTrigger = introduce_triggers
% trigger bits and codes for triggerInit and triggerSend
% below is the list of stim channels and their corresponding code
% since we are using dbs recording we only have triggers between 
% 1 and 31
% the more important triggers are being sent using one channel only
% for mor info check github wiki
% (https://github.com/tghafari/mTBI-predict/wiki/1.1.-MEG-and-Eyetracker-coding-schemes)

cfgTrigger.off = 0;
cfgTrigger.cueRight = 1; % start of attention orientation
cfgTrigger.cueLeft = 2; % start of attention orientation
cfgTrigger.trialStart = 3;
cfgTrigger.stimOnset = 4;  % onset of moving gratings- end of attention orientation
cfgTrigger.catchOnset = 5; % onset of catch trial
cfgTrigger.dotOnRight = 6;
cfgTrigger.dotOnLeft = 7;
cfgTrigger.resp = 8;  %  button press
cfgTrigger.blkStart = 20;  
cfgTrigger.blkEnd = 21; 
cfgTrigger.expEnd = 30; 
cfgTrigger.abort = 31;

end
