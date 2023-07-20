Betaski = 151;

X = [90 60 90;90 90 30;60 60 60;60 60 90;30 30 30];
[row,~] = size(X);
data = [ones(row,1) X];
r = randn(5,1)*10;
Y = 2.*data(:,1) + 4.*data(:,2) + 4.*data(:,3) + 8.*data(:,4) + r;
learningRate = 0.0001;
itnumber = 100000;

[B] = flinreg(X,Y,learningRate,itnumber)


function [B] = flinreg(X,Y,learningRate,itnumber)
[row,~] = size(X);
data = [ones(row,1) X];
B = (data.'*data)^-1 * data.'*Y;

for i = 1:itnumber
%     JB = sum((Y-data*B).^2)/row;
%     if 1 == sum(i == [2,5,10,20,50,100,200,500,1000,2000,5000,10000,20000,50000,100000])
%         abs(dB) >= abs(((Y-data*B).'*data/row).')
%     end
    dB = ((data*B-Y).'*data/row).';
    B = B - learningRate * dB;
%     if 1 == sum(i == [1,2,5,10,20,50,100,200,500,1000,2000,5000,10000,20000,50000,100000])
%         dB
%         B
%     end
end

% B
end
