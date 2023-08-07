function  cfgTrigger = introduce_triggers
% cfgTrigger = introduce_triggers
% trigger bits and codes for triggerInit and triggerSend
% below is the list of stim channels and their corresponding code:
% STI001 = 1;  STI002 = 2;  STI003 = 4;  STI004 = 8; 
% STI005 = 16; STI006 = 32; STI007 = 64; STI008 = 128;


cfgTrigger.off = 0;
cfgTrigger.trialStart = 1;
cfgTrigger.cueRight = 2;
cfgTrigger.cueLeft = 4;
cfgTrigger.stimOnset = 8;  % onset of moving gratings
cfgTrigger.dotOnRight = 16;
cfgTrigger.dotOnLeft = 32;
cfgTrigger.resp = 64;  %  button press
cfgTrigger.blkStart = 128;  
cfgTrigger.expEnd = 256; 
cfgTrigger.abort = 512;

end
