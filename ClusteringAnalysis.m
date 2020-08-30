%% pass the sequence through the network without updating weights
% record hidden unit activation

    contextt(1,:) = zeros(1,Nh); % initialize context input
for nn = 1:length(seq)
    
    % input from the sequence
    x = T(:,nn)'; 
    
    % grab context input
    c = zeros(1,Nh); 
    for j = 1:Nh
        c(j)=contextt(nn,j);
    end
    
    % input + context input
    I = [x c];    
    I = I'; 
    
    h=1./(1+exp(-v*I));% input & context to hidden unit activation
    y=1./(1+exp(-w*h));% hidden to output unit activation
    contextt(nn+1,:) = h; % copy current hidden output to context for use in next iteration
end

%%
hid1 = zeros(1,20);
hid2 = zeros(1,20);
hid3 = zeros(1,20);
hid4 = zeros(1,20);

for p = 1:length(seq)
    if seq(p) == 1
        hid1 = [hid1;contextt(p+1,:)];
    elseif seq(p) == 2
        hid2 = [hid2;contextt(p+1,:)];
    elseif seq(p) == 3
        hid3 = [hid3;contextt(p+1,:)];
    else
        hid4 = [hid4;contextt(p+1,:)];
    end
end

hid1 = hid1(2:end,:);
hid2 = hid2(2:end,:);
hid3 = hid3(2:end,:);
hid4 = hid4(2:end,:);

one = mean(hid1);
two = mean(hid2);
three = mean(hid3);
four = mean(hid4);

X = [one;two;three;four];
cluster = kmeans(X,2);

