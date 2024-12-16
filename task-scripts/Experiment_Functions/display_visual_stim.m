function cfgOutput = display_visual_stim(presentingStr, nstim, cfgScreen, cfgExp, cfgOutput, cfgStim, cfgTrigger, cfgEyelink)
% cfgOutput = display_visual_stim(presentingStr, nstim, cfgScreen, cfgExp, cfgOutput, cfgStim, cfgTrigger, cfgEyelink)
% draw ans flip visual stimulus with coordinates in cfgScreen
% for the duration specified in cfgExp

if cfgExp.corrResp(nstim)
    cfgOutput.stmOnTmPnt(nstim) = send_trigger(cfgTrigger, cfgExp, cfgTrigger.stimOnset, cfgEyelink...
        , 'stimulus onset');
    for frm = 1:cfgExp.stimFrm(nstim)
        Screen('DrawTextures', cfgScreen.window, [presentingStr.visStimR{nstim}{frm}, presentingStr.visStimL{nstim}{frm}]...
            , [], [cfgStim.destVisStimR; cfgStim.destVisStimL]');
        Screen('FillOval', cfgScreen.window, cfgScreen.fixDotColor, cfgScreen.fixDotRect);
        Screen('Flip', cfgScreen.window, cfgScreen.vbl + (cfgScreen.waitFrm - 0.5) * cfgScreen.ifi);
    end
    cfgOutput.dotOnTmPnt(nstim) = send_trigger(cfgTrigger, cfgExp, str2double(cfgTrigger.dotDir{nstim,1}), cfgEyelink...
        , sprintf('dot onset to %s', cfgTrigger.dotDir{nstim,2}));
    for frmDot = cfgExp.stimFrm(nstim):cfgExp.stimFrm(nstim) + cfgExp.dotFrm  % continue image presentation from previous for loop
        Screen('DrawTextures', cfgScreen.window, [presentingStr.visStimR{nstim}{frmDot}, presentingStr.visStimL{nstim}{frmDot}]...
            , [],[cfgStim.destVisStimR; cfgStim.destVisStimL]');
        Screen('FillOval', cfgScreen.window, [cfgScreen.fixDotColor, cfgScreen.fixDotFlashColor']...
            , [cfgScreen.fixDotRect, cfgStim.rectRL(cfgExp.trialMatrix(nstim),:)']);  % put the white dot according to the cue direction(cfgExp.trialMatrix-> 1:right, 2:left)
        Screen('Flip', cfgScreen.window, cfgScreen.vbl + (cfgScreen.waitFrm - 0.5) * cfgScreen.ifi);
    end
    cfgOutput.respStartTime(nstim) = GetSecs; % to get reaction times relative to stimulus offset
    Screen('FillOval', cfgScreen.window, cfgScreen.fixDotColor, cfgScreen.fixDotRect);
    Screen('Flip', cfgScreen.window, cfgScreen.vbl + (cfgScreen.waitFrm - 0.5) * cfgScreen.ifi);

else  % for catch trials
    cfgOutput.catchOnTmPnt(nstim) = send_trigger(cfgTrigger, cfgExp, cfgTrigger.catchOnset, cfgEyelink...
        , 'catch onset');
    for frm = 1:cfgExp.stimFrm(nstim)
        Screen('DrawTextures', cfgScreen.window, [presentingStr.visStimR{nstim}{frm}, presentingStr.visStimL{nstim}{frm}]...
            , [], [cfgStim.destVisStimR; cfgStim.destVisStimL]');
        Screen('FillOval', cfgScreen.window, cfgScreen.fixDotColor, cfgScreen.fixDotRect);
        Screen('Flip', cfgScreen.window, cfgScreen.vbl + (cfgScreen.waitFrm - 0.5) * cfgScreen.ifi);
    end
    Screen('FillOval', cfgScreen.window, cfgScreen.fixDotColor, cfgScreen.fixDotRect);
    Screen('Flip', cfgScreen.window, cfgScreen.vbl + (cfgScreen.waitFrm - 0.5) * cfgScreen.ifi);
    cfgOutput.respStartTime(nstim) = GetSecs; % to get reaction times relative to stimulus offset
    
end
end

