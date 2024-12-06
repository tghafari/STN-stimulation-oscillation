function timepoint = send_trigger(cfgTrigger, cfgExp, code, cfgEyelink, eyelinkMsg)
% timepoint = send_trigger(cfgTrigger, cfgExp, code, cfgEyelink, eyelinkMsg)
% sends trigger during MEG, code should indicate trigger code you want to
% send, eyelinkMsg includes the message you want to send to eyelink as
% trigger

outp(cfgTrigger.address1,code); % send trigger code, e.g., 16 (pin 5)
WaitSecs(0.005);  % wait 5ms to turn triggers off
outp(cfgTrigger.address1,0); % reset trigger port
WaitSecs(0.05);%ç­‰å¾…50msï¼Œä¹Ÿå°±æ˜¯LFPçš„Triggeræœ?5mså»¶è¿Ÿ

% get the time point of interest
outp(cfgTrigger.address2_send,0);

outp(cfgTrigger.address2_write,code);% è§¦å?‘ä½?æ¸…é›?è¿™æ­¥å?¯ä»¥ä¸?ç”¨
outp(cfgTrigger.address2_send,1); % å†™mark å€?  outp(cfgTrigger.address2_send,1);% è§¦å?‘ä½?ç½®1ï¼Œæ­¤æ—¶è§¦å?‘mark
WaitSecs(0.005);  % wait 5ms to turn triggers off


timepoint = GetSecs;  % get the time point of interest

end 