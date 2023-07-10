function cfgTrigger = initialise_trigger_port(cfgExp, cfgTrigger)
% cfgTrigger = initialise_trigger_port(cfgExp, cfgTrigger)
% initiates sending triggers to MEG pc and puts everything in cfgTrigger

cfgTrigger.handle = [];
cfgTrigger.address = [];
if cfgExp.EEGLab == 1
    cfgTrigger.address = hex2dec('378');  % port address
    cfgTrigger.handle = io64;
    cfgTrigger.status = io64(cfgTrigger.handle);
    io64(cfgTrigger.handle, cfgTrigger.address, 0);  % reset trigger
end

