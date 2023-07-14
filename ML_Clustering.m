Betaski = 151;

% rdata = randn(60,2)
% rdata_test = randn(60,2)
% 
% [index,C,sumd,D] = kmeans(rdata, 3)
% [index,C,borders] = fkmeans(rdata)
% indexx = tkmeans(rdata_test,C)

% rdata = [randn(40,2)+[-3 -2];randn(40,2)+[0 2];randn(40,2)+[3 -2]]
% rdata_test = [randn(40,2)+[-3 -2];randn(40,2)+[0 2];randn(40,2)+[3 -2]]
% 
% [mean,covariance,weight,pmax,cmax] = fgmm(rdata)
% [pmaxx,cmaxx] = tgmm(rdata_test,mean,covariance,weight)

% rdata = randn(10,2);
% 
% [records,log] = fhr(rdata)
% % thr nasıl olur bilemedim
% % slayt bir şey ekleyemedim

function [index,C,borders] = fkmeans(X)
% X = rdata;
itNumber = 100;
r = length(X(:,1));
c = length(X(1,:));
numCenter = 3;
centers = randn(c,numCenter); % init merkezler
centers2 = zeros(c,numCenter); %
cls = zeros(r+numCenter,numCenter); %
eoc = zeros(r,numCenter); %
clss = sym('cls',[r,numCenter]); %
borders = zeros(c+1,numCenter); % sınır değerleri
c = 1:numCenter; % sınıflar
for q = 1:itNumber
    for j = 1:numCenter
        k = mod(j,numCenter)+1;
        % k = c(j);
        midBorder = (centers(:,j) + centers(:,k))/2; % iki merkez ortasındaki nokta
        normBorder = centers(:,j) - centers(:,k); % iki merkezi birbirinie bağlayan doğrunun eğimi
        syms d real
        prob = midBorder.' * normBorder + d == 0; % merkezlerin ortasından geçen ve eğimini yukarıda bulduğumuz doğru dekleminin sabitine karar veriyoruz
        sol = double(solve(prob));
        
        %eq = normBorder.' * x + sol
        eq = @(x) normBorder.' * x + sol; % doğru deklemini bulduk
        % oluşturulan cls matrisinin ilk üç satırı sınıfları göstermek için
        % ayrılmıştı ve burada işaretleniyor
        cls(j,j) = eq(centers(:,j)) >= 0; % merkez işaretleniyor
        cls(k,j) = eq(centers(:,k)) >= 0; % merkez işaretleniyor
        for i = 1:r
            cls(i+numCenter,j) = eq(X(i,:).') >= 0; % değerler denkleme konduktan sonra elde edilen değerin pozitif olması sınıfa dahil olması anlamına gelir
        end
        if q == itNumber
            borders(:,j) = [sol;normBorder];
        end
    end
    for j = 1:numCenter
        k = mod(j,numCenter)+1;
        eoc(:,k) = cls(4:end,k) - cls(4:end,j); % soldaki noktaya daha yakın olması durumunda nötrlenmesi için soldaki değeri kendisinden çıkarıyoruz
    end
    for m = 1:numCenter
        idx = (eoc(:,m)==1); % sınıfa dahil değerlerin indxi
        clss(:,m) = c(m)*idx; % sınıf numarasını yerleştiriyoruz
        centers2(:,m) = (X(idx,:).' * ones(length(X(idx)),1)) / length(X(idx)); % yeni merkezlere karar veriliyor
    end
    centers = centers2;
end

index = double(clss * ones(numCenter,1)); % sınıf matrisi oluşturuluyor
C = centers2; % çıktı değişkenine atama
end

function index = tkmeans(X,C)
% X = rdata_test;
r = length(X(:,1));
numCenter = length(C(1,:));
cls = zeros(r+numCenter,numCenter); %
eoc = zeros(r,numCenter); %
clss = sym('cls',[r,numCenter]); %
c = 1:numCenter; % sınıflar
for j = 1:numCenter
    k = mod(j,numCenter)+1;
    % k = c(j);
    midBorder = (C(:,j) + C(:,k))/2; % iki merkez ortasındaki nokta
    normBorder = C(:,j) - C(:,k); % iki merkezi birbirinie bağlayan doğrunun eğimi
    syms d real
    prob = midBorder.' * normBorder + d == 0; % merkezlerin ortasından geçen ve eğimini yukarıda bulduğumuz doğru dekleminin sabitine karar veriyoruz
    sol = double(solve(prob));
    
    %eq = normBorder.' * x + sol
    eq = @(x) normBorder.' * x + sol; % doğru deklemini bulduk
    % oluşturulan cls matrisinin ilk üç satırı sınıfları göstermek için
    % ayrılmıştı ve burada işaretleniyor
    cls(j,j) = eq(C(:,j)) >= 0; % merkez işaretleniyor
    cls(k,j) = eq(C(:,k)) >= 0; % merkez işaretleniyor
    for i = 1:r
        cls(i+numCenter,j) = eq(X(i,:).') >= 0; % değerler denkleme konduktan sonra elde edilen değerin pozitif olması sınıfa dahil olması anlamına gelir
    end
end
for j = 1:numCenter
    k = mod(j,numCenter)+1;
    eoc(:,k) = cls(4:end,k) - cls(4:end,j); % soldaki noktaya daha yakın olması durumunda nötrlenmesi için soldaki değeri kendisinden çıkarıyoruz
end
for m = 1:numCenter
    idx = (eoc(:,m)==1); % sınıfa dahil değerlerin indxi
    clss(:,m) = c(m)*idx; % sınıf numarasını yerleştiriyoruz
end

index = double(clss * ones(numCenter,1)); % sınıf matrisi oluşturuluyor

end

function [mean,covariance,weight,pmax,cmax] = fgmm(X)
% X = rdata;
itNumber = 50;
r = length(X(:,1));
c = length(X(1,:));
numCenter = 3;
m = numCenter;

% başlangıç ortalama, kovaryans ve ağırlık değerleri belirlendi
mean = randn(numCenter,c); % 3x2
covariance = randn(c,c,numCenter); % 2x2x3
weight = ones(numCenter,1)*1/numCenter; % 3x1

% N(x|mean,cov,weight)
N = @(x,mean,covariance,weight,n,c) (exp(-0.5 * (x(n,:)-mean(c,:)) * covariance(:,:,c)^(-1) * (x(n,:)-mean(c,:)).')) / ((2 * pi)^(m/2) * sqrt(abs(det(covariance(:,:,c))))); %weight(c,1)
% weighted density function
wN = @(x,mean,covariance,weight,n,c) (weight(c,1) * N(x,mean,covariance,weight,n,c));
for q = 1:itNumber
    res = zeros(r,numCenter);
    ress = zeros(r,numCenter);

    % Expectation
    for a = 1:r
        for b = 1:numCenter
            res(a,b) = wN(X,mean,covariance,weight,a,b);
        end
        for b = 1:numCenter
            ress(a,b) = res(a,b)/sum(res(a,:));
        end
    end
    
    % Maximization
    weight = ress.'*ones(r,1)/r; % 3x1

    mean = zeros(numCenter,c);
    for i = 1:r
        mean = mean + (ress(i,:).'*X(i,:));
    end
    mean = mean./weight/r; %3x2

    covariance = zeros(c,c,numCenter); %2x2x3
    for j = 1:numCenter
        for i = 1:r
            covariance(:,:,j) = covariance(:,:,j) + ress(i,j)*(X(i,:)-mean(j,:)).'*(X(i,:)-mean(j,:));
        end
        covariance(:,:,j) = covariance(:,:,j)./weight(j,1)/r;
    end
end

clear max
[pmax, cmax] = max(ress,[],2);
end

function [pmax,cmax] = tgmm(X,mean,covariance,weight)
% X = rdata_test;
r = length(X(:,1));
numCenter = 3;

% N(x|mean,cov,weight)
N = @(x,mean,covariance,weight,n,c) (exp(-0.5 * (x(n,:)-mean(c,:)) * covariance(:,:,c)^(-1) * (x(n,:)-mean(c,:)).')) / ((2 * pi)^(numCenter/2) * sqrt(abs(det(covariance(:,:,c))))); %weight(c,1)
% weighted density function
wN = @(x,mean,covariance,weight,n,c) (weight(c,1) * N(x,mean,covariance,weight,n,c));

res = zeros(r,numCenter);
ress = zeros(r,numCenter);

% Expectation
for a = 1:r
    for b = 1:numCenter
        res(a,b) = wN(X,mean,covariance,weight,a,b);
    end
    for b = 1:numCenter
        ress(a,b) = res(a,b)/sum(res(a,:));
    end
end

clear max
[pmax, cmax] = max(ress,[],2);
end

function [records,log] = fhr(X)
% X = rdata3;
r = length(X(:,1));
c = length(X(1,:));
records = zeros(r-1,3);
weights = ones(r,1);
log = 1:r;

for k = 1:r-1
    d = zeros(r,r-1);
    for i = 1:r
        for j = i+1:r
            d(i,j) = sqrt(sum((X(i,:)-X(j,:)).^2)); % gözlemlerin arasındaki mesafeleri tablolanıyor
        end
    end
    min_val = min(d(d>0)); % min mesafe bulunuyor
    [r1 r2] = find(d==min_val); % min mesafedeki gözlem numaraları
    
    % bu gözlemlerden ilkler seçiliyor
    r1 = r1(1);
    r2 = r2(1);
    records(k,:) = [r1 r2 min_val]; % kayıt alınıyor
    
    % yeni gruplar ve ağırlıkları güncelleniyor
    X(r1,:) = (X(r1,:)*weights(r1,1)+X(r2,:)*weights(r2,1))/(weights(r1,1)+weights(r2,1));
    weights(r1,:) = weights(r1,1)+weights(r2,1);
    X(r2,:) = X(r1,:);
    weights(r2,:) = weights(r1,:);

    % birleşen iki grubun da tüm üyeleri birleşiyor
    [sze,~] = size(X(log(end,:)==r1,:));
    X(log(end,:)==r1,:) = ones(1,sze).' * X(r1,:);
    weights(log(end,:)==r1,:) = ones(1,sze).' * weights(r1,:);
    [sze2,~] = size(X(log(end,:)==r2,:));
    X(log(end,:)==r2,:) = ones(1,sze2).' * X(r1,:);
    weights(log(end,:)==r2,:) = ones(1,sze2).' * weights(r1,:);
    
    % log devam ettiriliyor
    temp = log(end,:);
    temp(temp==r2) = r1;
    log = [log;temp];
end
end
