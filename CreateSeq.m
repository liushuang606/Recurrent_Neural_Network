function [seq] = CreateSeq(L)
% Underlying pattern: PPGGP GGPPG GGGPP PPPGG
PZpattern = [1 1 0 0 1;0 0 1 1 0;0 0 0 1 1;1 1 1 0 0];
dim = size(PZpattern);
S = dim(1)*dim(2);
PZ = reshape(PZpattern',[S,1]);

% generate tone sequence
seq = zeros(1,L);

for i = 1:(L/S)
% Mandarin has 4 tones in its tone inventory
tone = zeros(1,S);
for n = 1:S
    if PZ(n) == 1
        tone(n)= randsample([3 4],1,true,[0.5 0.5]);
    else
        tone(n) = randsample([1 2],1,true,[0.7 0.3]);
    end
end
seq((1+S*(i-1)):(S*i)) = tone;
end
end
