function cfgTxt = txt_collection
% cfgTxt = txt_collection
% collection of all texts used in the experiment

cfgTxt.quitTxt = 'Are you sure you want to abort the experiment? y/n';
cfgTxt.startTxt = double('准备好后，请按Y键开始测试');%'Are you ready to start? \n Tell the experimenter to start the task when you are ready';
cfgTxt.breakTxt = double('准备好后，请按Y键继续测试');%'Take a break \n tell the experimenter to continue when you are ready';
cfgTxt.endTxt = double('恭喜你完成测试 :-)');
instr1 = double('最开始，会有一个白点出现在屏幕中央，请盯住它');%'First a dot appears at the centre of the screen. \n\n Please look at that dot for the duration of the task.' ;
instr2 = double('然后，白点下会出现一个箭头，请对箭头指向保持注意');%'Then an arrow will appear below the fixation point. \n\n The arrow shows the direction to which you should attend.';
instr3 = double('此后，你会白点两侧看到两个旋转的圆圈');%'Then you will see two circular moving gratings \n on the two sides of the screen. \n\n Please attend to the grating that was cued by the arrow \n without moving your eyes.';
instr4 = double('有一个红点会出现在箭头指过的圆圈中间；若是右边则按空格，若是左边则不做反应');%'A red dot will appear at the centre of the attended circle. \n\n Please press the right index finger as soon as you see the dot.';
cfgTxt.instrTxt = {instr1, instr2, instr3, instr4};

end