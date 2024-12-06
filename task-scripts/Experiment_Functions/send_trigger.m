function timepoint = send_trigger(cfgTrigger, cfgExp, code, cfgEyelink, eyelinkMsg)
% timepoint = send_trigger(cfgTrigger, cfgExp, code, cfgEyelink, eyelinkMsg)
% sends trigger during MEG, code should indicate trigger code you want to
% send, eyelinkMsg includes the message you want to send to eyelink as
% trigger

outp(cfgTrigger.address1,code); % send trigger code, e.g., 16 (pin 5)
WaitSecs(0.005);  % wait 5ms to turn triggers off
outp(cfgTrigger.address1,0); % reset trigger port
WaitSecs(0.05);%等待50ms，也就是LFP的Trigger�?5ms延迟

% get the time point of interest
outp(cfgTrigger.address2_send,0);

outp(cfgTrigger.address2_write,code);% 触�?��?清�?这步�?�以�?用
outp(cfgTrigger.address2_send,1); % 写mark �?  outp(cfgTrigger.address2_send,1);% 触�?��?置1，此时触�?�mark
WaitSecs(0.005);  % wait 5ms to turn triggers off


timepoint = GetSecs;  % get the time point of interest

end 