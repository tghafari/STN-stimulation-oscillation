function cfgTrigger = initialise_trigger_port(cfgExp, cfgTrigger)
% cfgTrigger = initialise_trigger_port(cfgExp, cfgTrigger)
% initiates sending triggers to MEG pc and puts everything in cfgTrigger
config_io;
global cogent;
if( cogent.io.status ~= 0 )
   error('inp/outp installation failed');
end
cfgTrigger.handle = [];
cfgTrigger.address = [];
cfgTrigger.address1 = hex2dec('CEFC');
cfgTrigger.address2_write = hex2dec('E8FC'); % port address
cfgTrigger.address2_send = hex2dec('E8FC')+2;
cfgTrigger.handle = io64;
cfgTrigger.status = io64(cfgTrigger.handle);
io64(cfgTrigger.handle, cfgTrigger.address1,0);  % reset trigger


