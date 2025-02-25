function cfgOutput = display_visual_stim(presentingStr, nstim, cfgScreen, cfgExp, cfgOutput, cfgStim, cfgTrigger, cfgEyelink)
% cfgOutput = display_visual_stim(presentingStr, nstim, cfgScreen, cfgExp, cfgOutput, cfgStim, cfgTrigger, cfgEyelink)
% draw ans flip visual stimulus with coordinates in cfgScreen
% for the duration specified in cfgExp

noResp = 1;
while noResp

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

        % Continue presenting the visual stimulus until the end of stim duration
        for fullStmFrm = cfgExp.stimFrm(nstim) + cfgExp.dotFrm:(cfgExp.fullStimFrm - cfgExp.stimFrm(nstim) + cfgExp.dotFrm)  % continue image presentation from previous for loop until the end of stim duration
            Screen('DrawTextures', cfgScreen.window, [presentingStr.visStimR{nstim}{frmDot}, presentingStr.visStimL{nstim}{frmDot}]...
                , [],[cfgStim.destVisStimR; cfgStim.destVisStimL]');
            Screen('FillOval', cfgScreen.window, cfgScreen.fixDotColor, cfgScreen.fixDotRect);
            Screen('Flip', cfgScreen.window, cfgScreen.vbl + (cfgScreen.waitFrm - 0.5) * cfgScreen.ifi);

            [presd, firstPrsd] = KbQueueCheck(cfgExp.deviceNum);  % listens for response
            keyCod = find(firstPrsd,1);  % collects the pressed key code

            if presd && ismember(keyCod, cfgExp.respKey)  % store response variables
                cfgOutput.respTmPnt(nstim) = send_trigger(cfgTrigger, cfgExp, cfgTrigger.resp, cfgEyelink...
                    , 'button press onset');
                % WaitSecs(0.002);  % wait to make sure the response trigger is reset
                cfgOutput.respTmKbQueue(nstim) = firstPrsd(keyCod);  % exact time of button press - more useful
                cfgOutput.keyName{nstim} = KbName(keyCod);  % which key was pressed
                cfgOutput.presd(nstim) = presd + 1;  % collect all responses for hit rate and correct rejection analysis
                cfgOutput.RT_KbQueue(nstim) = cfgOutput.respTmKbQueue(nstim) - cfgOutput.respStartTime(nstim);  % calculates RT - using time point in KbQueue
                if cfgExp.corrResp(nstim)
                    cfgOutput.RT_trig(nstim) = cfgOutput.respTmPnt(nstim) - cfgOutput.dotOnTmPnt(nstim);  % calculates RT - using triggers
                end
                KbQueueFlush;
                noResp = 0;
                break

            elseif presd && keyCod == cfgExp.quitKey
                Screen('Flip', cfgScreen.window);
                DrawFormattedText(cfgScreen.window, cfgTxt.quitTxt, 'center', 'center', [cfgScreen.white, cfgScreen.white, cfgScreen.white]);
                Screen('Flip', cfgScreen.window);
                [~, abrtPrsd] = KbStrokeWait;
                if abrtPrsd(cfgExp.yesKey)
                    cfgOutput.abrtTmPoint = send_trigger(cfgTrigger, cfgExp, cfgTrigger.abort, cfgEyelink...
                        , 'experiment aborted by user');  % send the quit trigger
                    cfgOutput = cleanup(cfgFile, cfgExp, cfgScreen, cfgEyelink, cfgOutput, cfgTrigger, cfgTxt, cfgStim);
                    warning('Experiment aborted by user')
                    break
                end
                KbQueueFlush;
            end
        end
            if ~presd
                cfgOutput.keyName{nstim} = 'no resp';
            end
            Screen('FillOval', cfgScreen.window, cfgScreen.fixDotColor, cfgScreen.fixDotRect);
            Screen('Flip', cfgScreen.window, cfgScreen.vbl + (cfgScreen.waitFrm - 0.5) * cfgScreen.ifi);
        

        if ~cfgExp.corrResp(nstim)  % for catch trials
            cfgOutput.catchOnTmPnt(nstim) = send_trigger(cfgTrigger, cfgExp, cfgTrigger.catchOnset, cfgEyelink...
                , 'catch onset');
            for frm = 1:cfgExp.stimFrm(nstim)
                Screen('DrawTextures', cfgScreen.window, [presentingStr.visStimR{nstim}{frm}, presentingStr.visStimL{nstim}{frm}]...
                    , [], [cfgStim.destVisStimR; cfgStim.destVisStimL]');
                Screen('FillOval', cfgScreen.window, cfgScreen.fixDotColor, cfgScreen.fixDotRect);
                Screen('Flip', cfgScreen.window, cfgScreen.vbl + (cfgScreen.waitFrm - 0.5) * cfgScreen.ifi);

                [presd, firstPrsd] = KbQueueCheck(cfgExp.deviceNum);  % listens for response
                keyCod = find(firstPrsd,1);  % collects the pressed key code

                if presd && ismember(keyCod, cfgExp.respKey)  % store response variables
                    cfgOutput.respTmPnt(nstim) = send_trigger(cfgTrigger, cfgExp, cfgTrigger.resp, cfgEyelink...
                        , 'button press onset');
                    % WaitSecs(0.002);  % wait to make sure the response trigger is reset
                    cfgOutput.respTmKbQueue(nstim) = firstPrsd(keyCod);  % exact time of button press - more useful
                    cfgOutput.keyName{nstim} = KbName(keyCod);  % which key was pressed
                    cfgOutput.presd(nstim) = presd + 1;  % collect all responses for hit rate and correct rejection analysis
                    cfgOutput.RT_KbQueue(nstim) = cfgOutput.respTmKbQueue(nstim) - cfgOutput.respStartTime(nstim);  % calculates RT - using time point in KbQueue
                    if cfgExp.corrResp(nstim)
                        cfgOutput.RT_trig(nstim) = cfgOutput.respTmPnt(nstim) - cfgOutput.dotOnTmPnt(nstim);  % calculates RT - using triggers
                    end
                    KbQueueFlush;
                    noResp = 0;
                    break
                elseif presd && keyCod == cfgExp.quitKey
                    Screen('Flip', cfgScreen.window);
                    DrawFormattedText(cfgScreen.window, cfgTxt.quitTxt, 'center', 'center', [cfgScreen.white, cfgScreen.white, cfgScreen.white]);
                    Screen('Flip', cfgScreen.window);
                    [~, abrtPrsd] = KbStrokeWait;
                    if abrtPrsd(cfgExp.yesKey)
                        cfgOutput.abrtTmPoint = send_trigger(cfgTrigger, cfgExp, cfgTrigger.abort, cfgEyelink...
                            , 'experiment aborted by user');  % send the quit trigger
                        cfgOutput = cleanup(cfgFile, cfgExp, cfgScreen, cfgEyelink, cfgOutput, cfgTrigger, cfgTxt, cfgStim);
                        warning('Experiment aborted by user')
                        break
                    end
                    KbQueueFlush;
                end
            end
        if ~presd
                    cfgOutput.keyName{nstim} = 'no resp';
        end

            Screen('FillOval', cfgScreen.window, cfgScreen.fixDotColor, cfgScreen.fixDotRect);
            Screen('Flip', cfgScreen.window, cfgScreen.vbl + (cfgScreen.waitFrm - 0.5) * cfgScreen.ifi);
            cfgOutput.respStartTime(nstim) = GetSecs; % to get reaction times relative to stimulus offset
        end
    end
    if (GetSecs - cfgOutput.respStartTime(nstim)) > ms2sec(cfgExp.respTimOut)  % stop listening after 1500msec
    KbQueueFlush;
    noResp = 0;
    break
    end
end

