function showInstructions(cfgScreen, cfgTxt, cfgExp, presentingStr, cfgStim)

Screen('Flip', cfgScreen.window);
DrawFormattedText(cfgScreen.window, cfgTxt.instrTxt{1}, 'center', 'center', [cfgScreen.white, cfgScreen.white, cfgScreen.white]);
Screen('Flip', cfgScreen.window);

notWaiting = false;  % only enable the experimenter to continue/ start
while ~notWaiting
    [~, contPresd] = KbStrokeWait(cfgExp.deviceNum);
    if contPresd(cfgExp.yesKey)
        notWaiting = true;
    end
    KbQueueFlush;
    WaitSecs(0.5)
end

DrawFormattedText(cfgScreen.window, cfgTxt.instrTxt{2}, 'center', 'center', [cfgScreen.white, cfgScreen.white, cfgScreen.white]);
Screen('Flip', cfgScreen.window);

notWaiting = false;  % only enable the experimenter to continue/ start
while ~notWaiting
    [~, contPresd] = KbStrokeWait(cfgExp.deviceNum);
    if contPresd(cfgExp.yesKey)
        notWaiting = true;
    end
    KbQueueFlush;
    WaitSecs(0.5)
end

DrawFormattedText(cfgScreen.window, cfgTxt.instrTxt{3}, 'center', 'center', [cfgScreen.white, cfgScreen.white, cfgScreen.white]);
Screen('Flip', cfgScreen.window);

notWaiting = false;  % only enable the experimenter to continue/ start
while ~notWaiting
    [~, contPresd] = KbStrokeWait(cfgExp.deviceNum);
    if contPresd(cfgExp.yesKey)
        notWaiting = true;
    end
    KbQueueFlush;
    WaitSecs(0.5)
end

DrawFormattedText(cfgScreen.window, cfgTxt.instrTxt{4}, 'center', 'center', [cfgScreen.white, cfgScreen.white, cfgScreen.white]);
Screen('Flip', cfgScreen.window, cfgScreen.vbl + (cfgScreen.waitFrm - 0.5) * cfgScreen.ifi);

notWaiting = false;  % only enable the experimenter to continue/ start
while ~notWaiting
    [~, contPresd] = KbStrokeWait(cfgExp.deviceNum);
    if contPresd(cfgExp.yesKey)
        notWaiting = true;
    end
    KbQueueFlush;
    WaitSecs(0.5)
end

stim = 2;
while cfgExp.trialMatrix(stim)>2; stim = stim + 1; end
        
for frm = 1:cfgExp.cueFrm*3
    Screen('DrawTexture', cfgScreen.window, presentingStr.cueStim{stim}, [], cfgStim.destCue);
    Screen('FillOval', cfgScreen.window, cfgScreen.fixDotColor, cfgScreen.fixDotRect);
    Screen('Flip', cfgScreen.window, cfgScreen.vbl + (cfgScreen.waitFrm - 0.5) * cfgScreen.ifi);
end
Screen('DrawTextures', cfgScreen.window, [presentingStr.visStimR{1}{2}, presentingStr.visStimL{2}{2}]...
    , [],[cfgStim.destVisStimR; cfgStim.destVisStimL]');
Screen('FillOval', cfgScreen.window, [cfgScreen.fixDotColor, cfgScreen.fixDotFlashColor']...
    , [cfgScreen.fixDotRect, cfgStim.rectRL(cfgExp.trialMatrix(stim),:)']);  % put the red dot according to the cue direction(cueRandIdx-> 1:left, 2:right)
Screen('Flip', cfgScreen.window, cfgScreen.vbl + (cfgScreen.waitFrm - 0.5) * cfgScreen.ifi);

notWaiting = false;  % only enable the experimenter to continue/ start
while ~notWaiting
    [~, contPresd] = KbStrokeWait(cfgExp.deviceNum);
    if contPresd(cfgExp.yesKey)
        notWaiting = true;
    end
    KbQueueFlush;
    WaitSecs(0.5)
end
