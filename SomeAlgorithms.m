Betaski = 151;

% data = [90 60 90;90 90 30;60 60 60;60 60 90;30 30 30];
% 
% e = pdist(data,"euclidean")
% d = apwdist(data)

% rdata = randn(20,2) * 15 + 100
% 
% Z = stanley(rdata)
% N = normy(rdata)

% data = [90 60 90;90 90 30;60 60 60;60 60 90;30 30 30]
% data_test = randn(5,3)*45;
% 
% [coeff,score,~,~,explained] = pca(data)
% [coeff2,score2,explained2] = fpca(data)
% % score hesabında problem var
% scores = tpca(data_test,coeff2)

% rdata = randn(80,2);
% kfold = 10;
% 
% folds = crssvalid(rdata,kfold)

% Y = randn(20,1) >= 0;
% Y_pred = rand(20,1);% >= 0.5;
r = 3;
e = rand(20,1);
Y = floor(e*r)
ee = rand(20,r);
Y_pred = ee./(ee*ones(r,1))

[Loss] = classLoss(Y,Y_pred,'log')

Y = randn(20,1) >= 0;
Y_pred = rand(20,1);

[Loss1] = classLoss(Y,Y_pred,'hinge')

function d = pairwise_distance(X)
% X = data;
r = length(X(:,1));
c = length(X(1,:));
d = zeros(1,r*(r-1)/2);
n = 1;
for i = 1:r
    for j = i+1:r
        d(1,n) = sqrt(sum((X(i,:)-X(j,:)).^2));
        n = n+1;
    end
end
end

function [Z] = stanley(X)
% X = rdata;

[row,col] = size(X);
mean = X.' * ones(row,1) / row;
covariance = cov(X);

ortX = (X.' - mean).';
Z = zeros(row,col);
for i = 1:col
    Z(:,i) = ortX(:,i).' / sqrt(covariance(i,i));
end
end

function [N] = normy(X)
% X = rdata;

[row,col] = size(X);

clear max
[~,xmax] = max(X,[],1);
clear min
[~,xmin] = min(X,[],1);

N = zeros(row,col);
for i = 1:col
%     clear max
%     [~,xmax] = max(X(:,i),[],1);
%     clear min
%     [~,xmin] = min(X(:,i),[],1);
    N(:,i) = (X(:,i) - X(xmin(i),i)) / (X(xmax(i),i) - X(xmin(i),i));
end
end

function [coeff,score,explained] = fpca(X)
% X = data;
r = length(X(:,1)); % sütun sayısı
c = length(X(1,:)); % kolon sayısı
covariance = cov(X); % kovaryans matrisi %(X-ones(r,r)*X/r).'*(X-ones(r,r)*X/r)/r;

syms lambda real
eq = det(covariance-lambda*eye(c)); % köklerinden özdeğer elde edeceğimiz denklem
lambda = double(vpa(root(eq,lambda))); % Eigenvalue
[coeff,~] = eig(covariance); % Eigenvectors

score = zeros(r,c);
for i = 1:r
    for j = 1:c
        score(i,j) = coeff(:,j).' * X(i,:).'; % gözlemlerin yeni bileşenler cinsinden ifadesi
    end
end
% score hesabında kullanılan coeff satır kare toplamı da kolon kare
% toplamına eşit çıkıyor

explained = lambda/sum(lambda)*100; % varyans yüzdeleri
end

function scores = tpca(X,coeff)
% X = data_train;
r = length(X(:,1)); % sütun sayısı
c = length(X(1,:)); % kolon sayısı

scores = zeros(r,c);
for i = 1:r
    for j = 1:c
        scores(i,j) = coeff(:,j).' * X(i,:).'; % gözlemlerin yeni bileşenler cinsinden ifadesi
    end
end
end

function folds = crssvalid(X,kfold)
% X = rdata;
% kfold = 10;
[row,~] = size(X);

c = mod(row,kfold);
if c == 0
    folds = zeros(row/kfold,kfold);
    for i = 1:kfold
        folds(:,i) = (i-1)*row/kfold + (1:row/kfold).';
    end
end
if c ~= 0
    folds = zeros((row-c)/kfold+1,kfold);
    for i = 1:kfold
        folds(:,i) = (i-1)*((row-c)/kfold+1) + (1:((row-c)/kfold+1)).';
    end
    folds(folds>row) = 0;
end
end

function [Loss] = classLoss(Y,Y_pred,type)
[row,col] = size(Y_pred);

Lm = 0;

if prod(type == 'log',"all") == 1
    for i = 1:col
        Lm = sum(Lm + (Y==i-1).*log(Y_pred(:,i)));
    end
end

if prod(type == 'hinge',"all") == 1
    % Loss = sum(max(0,(Y==1)-Y.*Y_pred));
    for i = 1:col
        Lm = sum(max(0,(Y==i-1)-(Y==i-1).*Y_pred(:,i)));
    end
end

Loss = -Lm/row;

end
