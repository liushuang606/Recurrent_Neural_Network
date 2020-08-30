% TermProject_SL.m
%
% This script is part of the term project for Psych 5618
% "Introduction to Computational Cognitive Neuroscience".
%
% To publish the script as a PDF (rather than the default HTML),
% type the following in Matlab's command window:
%
% publish('TermProject_SL','pdf')
%
% Submitted by Shuang Liu
%% Clear Matlab's workspace and print the name of this script

fprintf('\n%s \n', datestr(now)) ;    % print current time
fprintf('Executing %s ...\n', which(mfilename())) ; % name of this script

close all;   % close all figure windows, if any
clear all;   % erase all variables from the workspace
%% Student name

student_name = 'Shuang Liu' %#ok<NOPTS>
%% Generate training tone patterns
lt = 5000; % length of tone sequence
seq = CreateSeq(lt);
pass = 20; % number of passes
for n = 1:pass
Train(:,1+lt*(n-1):lt*n) = seq;
end

L = length(Train);

T = zeros(4,L);

for t = 1:L
    if Train(t) == 1
        T(1,t) = 1;
    elseif Train(t) == 2
        T(2,t) = 1;
    elseif Train(t) == 3
        T(3,t) = 1;
    else
        T(4,t) = 1;
    end
end

%% network structure
 Ni = 4; % number of input unit
 Nh = 20; % number of hidden units
 No = 4; % number of output unit
 Nc = Nh;% number of context units 
 
 % initialize weight
 v = rand(Nh,(Ni+Nc))-0.5; % weight between input and hidden AND weight between context and hidden
 w = rand(No,Nh)-0.5; % weight between hidden and ouput
 
 %% pass the sequence through the network
    context(1,:) = zeros(1,Nh); % initialize context input
for nn = 2:L
    
    % input from the sequence
    x = T(:,nn-1)'; 
    
    % grab context input
    c = zeros(1,Nh); 
    for j = 1:Nh
        c(j)=context(nn-1,j);
    end
    
    % input + context input
    I = [x c];    
    I = I'; 
    
    h=1./(1+exp(-v*I));% input & context to hidden unit activation
    y=1./(1+exp(-w*h));% hidden to output unit activation
    context(nn,:) = h; % copy current hidden output to context for use in next iteration
    
    yteach = T(:,nn); % teach signal
    
    werror = (y.*(1-y)).*(yteach-y);% teaching signal for updating the hidden layer to output weights
                          % errors between teaching and output activations
    w = w+0.3*(h*werror')';
    %if rand < 0.8; w = w+0.7*(h*werror')';% update hidden-to-output weights
    %else w = w+0.2*(rand(No,Nh)-0.5);end;
    
    verror = (h.*(1-h)).*(w'*werror);% teaching signal for updating the input to hidden layer weights
                         % backpropagated error from output layer to hidden layer
    v = v+0.3*(I*verror')';
    %if rand < 0.8; v = v+0.7*(I*verror')';% update input-to-hidden weights
    %else v=v+0.2*(rand(Nh,(Ni+Nc))-0.5);end;
    dglobal(nn-1) = sqrt(sum((yteach-y).^2)/4); % global RMS error
    % d(:,nn-1) = yteach-y; % local error
end

for k=1:40
dd(k)=mean(dglobal(k+(L-10001):40:(L-1)));
end

plot((1:1:40)',dd,'-o','LineWidth',2);axis([0 41 0.3 0.6]);

