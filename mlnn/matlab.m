%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% KCR Demo %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Omer Lang, olang@ucsd.edu

clc
close all
clear all

rand('twister',45)

%% Parameters

%%%%%%%%%%%%%%%%%%%%%%%% Data and reference points %%%%%%%%%%%%%%%%%%%%%%%%

% Data source: 0) load data used in previous experiment, 1) hand click data,
% 2) gaussian clusters, 3) disc and annulus, 4) checkerboard, 5) load data
% from txt file
data = 5;

% Data file: 1) Wine, 2) Iris, 3) Iono, 4) Bala, 5) Segm, 6) Sona, 7) Wdbc,
% 8) Glas, 9) Pima, 10) Soyb, 11) Yeas, 12) Germ
file = 1;

% Maximum number if data points
maxn = inf;

% Training/test split
train = .7;

% Toggles normalization of data to zero mean and unit variance:
% 0) normalize off, 1) normalize on
normalize = 1;

% Reference point method: 0) all data points, 1) randomly selected data
% points, 2) k-means, 3) maximum distance, 4) grid, 5) uniform
% Need only be specified if nonlinear = 1
reference = 0;

% Number of reference points
% Need only be specified if nonlinear = 1 and reference ~= 0
r = 100;

% Kernel width parameter
% Setting sigma2 = 0 causes sigma2 to be chosen using entropy heuristic
% Setting sigma2 = -1 causes sigma2 to be chosen using distance heuristic
% Need only be specified if nonlinear = 1
sigma2 = 8;

% KNN parameter
k = 3;

%%%%%%%%%%%%%%%%%%%%%%%%%%% Model parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Toggles regularization: 0) regularization off, 1) 1-norm regularization,
% 2) 2-norm regularization
regularization = 2;

% Toggles linear/nonlinear implementation: 0) linear, 1) nonlinear 
nonlinear = 1;

% Toggles diagonal/full matrix implementation: 0) full, 1) diagonal
diagonal = 0;

% Toggles convex/nonconvex implementation: 0) nonconvex, 1) convex
convex = 1;

% Toggles whether of not a metric is learning: 0) non-metric, 1) metric
% Need only be specified if convex = 1
metric = 1;

% Toggles projection onto set of matrices whos rows and colums sum to 0:
% 0) projection off, 1) projection on
% Need only be specified if nonlinear = 1
zerosum = 1;

% Inner loss function: 1) hinge, 2) smooth hinge, 3) quadratic, 4) logistic
loss1 = 3;

% Outer loss function: 0) none 1) hinge, 2) smooth hinge, 3) quadratic,
% 4) logistic
loss2 = 3;

% Epsilon calcuation mode: 0) single epsilon for entire space,
% 1) different epsilon for each training point, 2) affine function of point
emode = 0;

% Slack weighting parameter
c = .001;

% Desired dimensionality of the learned metric space
% Setting d = 0 causes L to be a square matrix
% Need only be specified if convex = 0
d = 2;

% Neighborhood conditioning factor
% Setting s = 0 causes s to equal one less than the number of classes
s = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%% Stopping criteria %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Size of frame used to check for convergence
frame = 10;

% Convergence threshold
convthresh = 0;

% Maximum number of steps
maxsteps = 10000;

% Maximum amount of time in seconds
maxtime = 30;

% Maximum number of backtracks
maxbackcount = 50;

%%%%%%%%%%%%%%%%%%%%%%%%% Algorithm performance %%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initial learning rate
lambda0 = .001;

% How much to increase lambda after each iteration
lambdaup = ((1+sqrt(5))/2)^(1/10);

% How much to decrease lambda if lambda is too large
lambdadown = ((1+sqrt(5))/2)^-1;

% Armijo rule number
armijo = .000001;

% Epsilon learning rate scaling factor
escale = 10;

%%%%%%%%%%%%%%%%%%%%%%% Figures and diagnostics %%%%%%%%%%%%%%%%%%%%%%%%%%%

% Figure mode: 0) figures off, 1) figures on
figures = 1;

% Toggle diognostics: 0) diognostics off, 1) diognostics on
diagnostics = 0;

% Learning curve: 0) learning curve off, 1) learning curve on
learningcurve = 0;

% Learning curve interval
% Need only be specified if learningcurve = 1
interval = 1;

if regularization == 0
    c = 1;
end

if diagonal
    zerosum = 0;
    convex = 1;
    d = 0;
end

%% Get Data

% Load or generate data (X) and class labels (Y).

if data == 0 %Load data from file
    load Data
elseif data == 1 %Hand click data
    P = [20 20 0 0]; %Number of points in each class
    W = [0 10 0 10]; %Window dimensions

    figure
    hold on
    axis(W)
    title(sprintf('n_1 = %d, n_2 = %d, n_3 = %d, n_4 = %d',P(1),P(2),P(3),P(4)))

    Y = [1*ones(P(1),1); 2*ones(P(2),1); 3*ones(P(3),1); 4*ones(P(4),1)];

    n = length(Y);

    X = zeros(n,2);

    for i = 1:n
        x = ginput(1);

        X(i,:) = x;

        if Y(i) == 1
            plot(x(1),x(2),'bs')
            P(1) = P(1)-1;        
        elseif Y(i) == 2
            plot(x(1),x(2),'go')
            P(2) = P(2)-1;
        elseif Y(i) == 3
            plot(x(1),x(2),'rd')
            P(3) = P(3)-1;
        else
            plot(x(1),x(2),'cd')
            P(4) = P(4)-1;
        end

        title(sprintf('n_1 = %d, n_2 = %d, n_3 = %d, n_4 = %d',P(1),P(2),P(3),P(4)))
    end

    close

    save Data X Y
elseif data == 2 %Generate up to four gaussian clusters
    % guassians(n1,mux,muy,sigma2x,sigma2y,rotation,n2,...)
    [X Y] = gaussians(50,-5,1,2,4,pi/8,...
                      50,4,4,2,2,5*pi/8,...
                      50,3,-3,2,3,-pi/4,...
                      50,-1,-5,2,2,3*pi/8);

    save Data X Y
elseif data == 3 %Generate disc and annulus
    % discannulus(n1,n2,r1,r2,r3,r4,sigma2); sigma2 is variance of WGN
    [X Y] = discannulus(25,25,4.75,4.75,5.25,5.25,0);

    save Data X Y
elseif data == 4 %Generate checkerboard
    % checker(n,prior,dimy,scaley,dimx,scalex)
    [X Y] = checker(108,5/9,3,3,3,3);

    save Data X Y
elseif data == 5 %Load data from txt file
    if file == 1
        load Wine
    elseif file == 2
        load Iris
    elseif file == 3
        load Iono
    elseif file == 4
        load Bala
    elseif file == 5
        load Segm
    elseif file == 6
        load Sona
    elseif file == 7
        load Wdbc
    elseif file == 8
        load Glas
    elseif file == 9
        load Pima
    elseif file == 10
        load Soyb
    elseif file == 11
        load Yeas
    elseif file == 12
        load Germ
    end

    Y = Y+(1-min(Y)); %#ok<NODEF>
end

[n D] = size(X); %n = number of points, D = dimension of input space

classes = unique(Y);

% Randomize data.
I = randperm(n);
X = X(I,:);
Y = Y(I);

if n > maxn
    X = X(1:maxn,:);
    Y = Y(1:maxn);
    n = maxn;
end

% Split into training and testing sets
numTest = round((1-train)*n);
n = n-numTest;
Xtest = X(n+1:end,:);
Ytest = Y(n+1:end);
X = X(1:n,:);
Y = Y(1:n);

% Sort data by class.
[Y I] = sort(Y);
X = X(I,:);

[Ytest I] = sort(Ytest);
Xtest = Xtest(I,:);

% if normalize
%     means = mean(X);
%     stds = std(X);
%     
%     X = bsxfun(@rdivide,bsxfun(@minus,X,means),stds);
%     Xtest = bsxfun(@rdivide,bsxfun(@minus,Xtest,means),stds);
% end

%% Analyze Original Data

% Calculate LOOE with KNN using euclidean distance. Diplay data along with
% incorrectly classifide data points (solid markers) and reference points (+).
% Use PCA if D > 2.

G = X*X';

G = bsxfun(@plus,diag(G),diag(G)')-2*G;

[G I] = sort(G,2);
B = Y(I(:,2:k+1));
[M R] = mode(B,2);
M(R == 1) = B(R == 1,1);
W = Y ~= M;
YY = Y+4*W;
LOOE1 = sum(W)/n;

% Calculate test error with KNN using euclidean distance.
if numTest > 0
    Gxtest = zeros(numTest,1);
    
    for i = 1:numTest
        Gxtest(i) = Xtest(i,:)*Xtest(i,:)';
    end
    
    Gx = zeros(1,n);
    
    for i = 1:n
        Gx(i) = X(i,:)*X(i,:)';
    end
    
    G = bsxfun(@plus,Gxtest,Gx)-2*Xtest*X';

    [G I] = sort(G,2);
    B = Y(I(:,1:k));
    [M R] = mode(B,2);
    M(R == 1) = B(R == 1,1);
    W = Ytest ~= M;
    TEST1 = sum(W)/numTest;
end

X0 = X;

if normalize
    means = mean(X);
    stds = std(X);
    
    X = bsxfun(@rdivide,bsxfun(@minus,X,means),stds);
    Xtest = bsxfun(@rdivide,bsxfun(@minus,Xtest,means),stds);
end

if nonlinear

%% Get Reference Points

    % Choose reference points (Z).

    if reference == 0 %Use original data points
        Z = X;

        r = n;
    elseif reference == 1 %Choose r random data points
        I = randperm(n);

        Z = X(I(1:r),:);
    elseif reference == 2 %Use kmeans to find r centroids
        [Dummy Z] = kmeans(X,r,'EmptyAction','singleton');
    elseif reference == 3 %Pick r data points that are from each other
        Z = zeros(r,size(X,2));

        G = zeros(1,n);

        q = ceil(rand*n);

        Z(1,:) = X(q,:);

        G(q) = -inf;

        for i = 2:r
            for j = 1:n
                G(j) = G(j)+ norm(Z(i-1,:)-X(j,:));
            end

            [dummy q] = max(G);

            Z(i,:) = X(q,:);

            G(q) = -inf;
        end
    elseif reference == 4 %Generate a grid of round(r^(1/D)) points
        p = round(r^(1/D));

        r = p^D;

        Z = [];

        for i = 1:D
            Temp1 = linspace(min(X(:,i)),max(X(:,i)),p);
            Temp2 = ones(p^(i-1),1)*(1:p);

            Z = [repmat(Z,p,1) Temp1(Temp2(:))'];
        end
    elseif reference == 5 %Generate r points uniformly at random from the range of X
        Z = rand(r,D).*(ones(r,1)*(max(X)-min(X)))+ones(r,1)*min(X);
    end

%% Get Sigma2 Heuristic

    % Find the sigma2 which minimizes the KL divergence between the
    % distribution of points in the kernel space and some desired distribution.

    if sigma2 <= 0
        Gz = zeros(r,1);

        for i = 1:r
            Gz(i) = Z(i,:)*Z(i,:)';
        end

        Gx = zeros(1,n);

        for i = 1:n
            Gx(i) = X(i,:)*X(i,:)';
        end

        G = bsxfun(@plus,Gz,Gx)-2*Z*X';

        if sigma2 == -1
            sigma2 = 2*sum(sum(G))/(r*n);
        else
            sigma2 = 10.^(-2:.01:6);
            t = length(sigma2);

            p = r*n;
            nbins = ceil(log2(p)+1);

            Q = ones(nbins,1)/nbins;
            %Q = linspace(1,0,nbins+2)'; Q = Q(2:end-1); Q = Q/sum(Q);

            KL = zeros(1,t);

            bins = linspace(0,1,nbins+1);

            for i = 1:t
                K = exp(G/(-2*sigma2(i)));

                N = histc(K(:),bins);

                P = N(1:end-1)/p;

                KL(i) = -sum(P.*log2(P./Q));
            end

            figure
            plot(log10(sigma2),KL)
            title('KL Divergence Between Actual and Desired Distributions')

            [dummy i] = max(KL);

            sigma2 = sigma2(i);

            K = exp(G/(-2*sigma2));

            N = histc(K(:),bins);

            P = N(1:end-1)/p;

            figure
            hold on
            bar(bins(1:end-1)+(bins(2)-bins(1))/2,P)
            plot(bins(1:end-1)+(bins(2)-bins(1))/2,Q)
            title('Distribution of Points in the Kenrel Space')
        end
    end
else
    r = D;
end

%% Display Data

% If dimensionality of data is greater than 2, perform PCA.
if D > 2
    Xc = bsxfun(@minus,X0,mean(X0,1));

    S = zeros(D);

    for i = 1:n
        S = S+Xc(i,:)'*Xc(i,:);
    end

    [V J] = eig(S);
    [J I] = sort(abs(diag(J)),'descend');
    U = V(:,[I(1) I(2)]);
    U = [sign(sum(U(:,1)))*U(:,1) sign(sum(U(:,2)))*U(:,2)];

    Xp = X0*U;
    if nonlinear && reference
        Zp = Z*U;
    end
else
    Xp = X0;
    if nonlinear && reference
        Zp = Z;
    end
end

figure
plot(Xp(YY == 1,1),Xp(YY == 1,2),'bs',Xp(YY == 2,1),Xp(YY == 2,2),'go',...
     Xp(YY == 3,1),Xp(YY == 3,2),'rd',Xp(YY == 4,1),Xp(YY == 4,2),'cd')
hold on
plot(Xp(YY == 5,1),Xp(YY == 5,2),'bs','MarkerFaceColor','b')
plot(Xp(YY == 6,1),Xp(YY == 6,2),'go','MarkerFaceColor','g')
plot(Xp(YY == 7,1),Xp(YY == 7,2),'rd','MarkerFaceColor','r')
plot(Xp(YY == 8,1),Xp(YY == 8,2),'cd','MarkerFaceColor','c')
if nonlinear && reference
    plot(Zp(:,1),Zp(:,2),'+k')
end
axis equal
hold off
title(sprintf('Original Data, X, n = %d, D = %d, LOOE = %g%%',n,D,LOOE1*100))

%% Calculate and Display Algorithm Inputs

if nonlinear
    % Kx is a r x n kernel matrix where Kx(i,j) = K(Z(i,:),X(j,:)).

    Gx = zeros(n,1);

    for i = 1:n
        Gx(i) = X(i,:)*X(i,:)';
    end

    Gz = zeros(1,r);

    for i = 1:r
        Gz(i) = Z(i,:)*Z(i,:)';
    end

    G = bsxfun(@plus,Gx,Gz)-2*X*Z';

    Kx = exp(G/(-2*sigma2));
    
    Gxtest = zeros(numTest,1);

    for i = 1:numTest
        Gxtest(i) = Xtest(i,:)*Xtest(i,:)';
    end

    G = bsxfun(@plus,Gxtest,Gz)-2*Xtest*Z';

    Kxtest = exp(G/(-2*sigma2));

    figure
    imshow(-Kx,[-1 0])
    title(['Guassian Kernel Matrix, Data, K_x, ' '\sigma^2 = ' sprintf('%.3g',sigma2)])

    % Kz is a r x r kernel matrix where Kz(i,j) = K(Z(i,:),Z(j,:)).

    if reference
        G = Z*Z';

        G = bsxfun(@plus,diag(G),diag(G)')-2*G;

        Kz = exp(G/(-2*sigma2));

        figure
        imshow(-Kz,[-1 0])
        title(['Guassian Kernel Matrix, Reference, K_z, ' '\sigma^2 = ' sprintf('%.3g',sigma2)])
    else
        Kz = Kx;
    end
else
    Kx = X;
    Kxtest = Xtest;
    Kz = eye(r);
end

% Same class matrix (T). T(i,j) = 1 if Y(i) = Y(j). Otherwise, T(i,j) = -1.

if s == 0
    s = length(unique(Y))-1;
end

B = bsxfun(@ne,Y,Y');

S = repmat(s,n);
S(B) = 1;

T = ones(n);
T(B) = -1;

% figure
% imshow(-T,[-1 1])
% title('Same Class Matrix, \tau')

drawnow

Hr = eye(r)-ones(r)/r; %Centering matrix of size D x D

%% Initialize L and e

if diagonal
    L = zeros(r,1);
elseif convex
    L = zeros(r);
else
    % If nonlinear, initialize L using KPCA.
    % If linear, initialize L using PCA.

    if nonlinear
        if d == 0
            d = r;
        end
        
        Om = ones(r)/r;

        Kc = Kz-Om*Kz-Kz*Om+Om*Kz*Om;

        [V J] = eig(Kc);

        [J I] = sort(diag(J),'descend');

        V = V(:,I);

        L = V(:,1:d)';
    else
        if d == 0
            d = D;
        end
        
        Xc = bsxfun(@minus,X,mean(X));

        [V J] = eig(Xc'*Xc/n);

        [J I] = sort(diag(J),'descend');

        V = V(:,I);

        L = V(:,1:d)';
    end

    if loss1 == 4
        C = L'*L;

        A = Kx*C*Kx';

        G = bsxfun(@plus,diag(A),diag(A)')-2*A;

        L = sqrt(-log(.5)/mean(mean(G)))*L;
    end

    if zerosum && nonlinear
        L = L*Hr;
    end
end

if emode == 0
    e = 0; %e is the kernel width being learned
elseif emode == 1
    e = zeros(n,1); %e is the vector of kernel widths being learned, one for each training point
elseif emode == 2
    e = zeros(r+1,1); %e is a vector such the kernel width for a point k is e'*[k 1]
end

%% Perform Gradient Descent

% Perform gradient descent until one of the stopping criteria is met.

% Initialize variables.

stepcount = -1; %Number of steps taken along the gradient
done = 0; %Indicates if stopping criteria has been met
backcount = 0; %Number of consecutive backtracks
lambda = lambda0; %Learning rate
fold = inf; %Value of the objective function at previous iteration
dfdL = zeros(size(L)); %Gradient of L
dfde = zeros(size(e)); %Gradient of e
F = zeros(1,maxsteps+1); %Keeps track of objective function values

N = s*(sum(bsxfun(@eq,Y,Y'),2)-1); %N(i) = # of points with the same label as point i

On = ones(n,1); %Ones column vector of length n
Ir = eye(r); %Identity matrix of size r x r
Or = ones(r,1); %Ones column vector of length r

count1 = 0; %Counts the number of times f is calculated
count2 = 0; %Counts the numner of times dfdL is calculated
count3 = 0; %Counts the number of times the eigenvalue decompostion of L is calculated
count4 = 0; %Counts the number of times the the zerosum projection is calculated

if diagnostics
    Time = zeros(1,maxsteps+1);
    Numactive1 = zeros(1,maxsteps+1);
    Numactive2 = zeros(1,maxsteps+1);
    Numactive3 = zeros(1,maxsteps+1);
    Reg = zeros(1,maxsteps+1);
    E = zeros(1,maxsteps+1);
    Lambda = zeros(1,maxsteps+1);
    Backcount = zeros(1,maxsteps+1);
    DfdLnorm = zeros(1,maxsteps+1);
    Dfde = zeros(1,maxsteps+1);
end

if learningcurve
    trainCurve = zeros(1,floor(maxsteps/interval)+1);
    testCurve = zeros(1,floor(maxsteps/interval)+1);
    learningTime = zeros(1,floor(maxsteps/interval)+1);
end

if figures
    display('Press any key to continue.')       
    pause

    figure
end

t0 = tic; %Starts clock

while ~done
    if ~diagonal
        if convex
            C = L; %C parameterizes a mahalanobis distance between points
        else
            C = L'*L;
        end
    end

    % Calculate regularization term.
    if regularization == 1 %1-norm regularization
        if nonlinear
            if diagonal
                reg = L'*diag(Kz);
            else
                reg = C(:)'*Kz(:); %Value of the regularization term
            end
        else
            if diagonal
                reg = sum(L);
            else
                reg = trace(C);
            end
        end
    elseif regularization == 2 %2-norm regularization
        if nonlinear
            if diagonal
                Temp1 = bsxfun(@times,L,Kz);
                Temp2 = Temp1';
                reg = .5*Temp1(:)'*Temp2(:);
            else
                Temp1 = C*Kz;
                Temp2 = Temp1';
                reg = .5*Temp1(:)'*Temp2(:);
            end
        else
            if diagonal
                reg = .5*L'*L;
            else
                reg = .5*C(:)'*C(:);
            end
        end
    else %no regularization
        reg = 0;
    end

    % Calculate the value of epsilon for each training point.
    if emode == 0
        ei = repmat(e,n,1);
    elseif emode == 1
        ei = e;
    elseif emode == 2
        ei = [Kx On]*e;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if diagonal
        Temp = bsxfun(@times,Kx,sqrt(L)');
        A = Temp*Temp';
    else
        A = Kx*C*Kx'; %Weighted inner product matrix
    end
    
    G = bsxfun(@plus,diag(A),diag(A)')-2*A;
    
    if loss1 < 4
        O = max(1+T.*bsxfun(@minus,G,ei),0); %Argument of inner hinge function for each point pair
    else
        O = T.*bsxfun(@minus,G,ei);
    end
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Calculate slack penalty.
    if loss1 == 1 %hinge loss
        H1 = O; %Value of inner hinge function for each point pair
    elseif loss1 == 2 %smooth hinge loss
        H1 = .5.*O.^2;
        
        O1 = O > 1;
        
        H1(O1) = O(O1)-.5;
    elseif loss1 == 3 %quadratic loss
        H1 = O.^2;
    else %logistic loss
        H1 = O;
        
        O2 = O < 37;
        
        H1(O2) = log(1+exp(O(O2)));
    end
    
    if s ~= 1
        H1 = S.*H1;
    end
    
    if loss2 == 0
        H2 = sum(H1,2);
    else
        if loss2 < 4
            Q = max(1+sum(H1,2)-N,0);
        else
            Q = sum(H1,2)-N;
        end

        if loss2 == 1 %hinge loss
            H2 = Q; %Value of outer hinge function for each point
        elseif loss2 == 2 %smooth hinge loss
            H2 = .5.*Q.^2;

            Q1 = Q > 1;

            H2(Q1) = Q(Q1)-.5;
        elseif loss2 == 3 %quadratic loss
            H2 = Q.^2;
        else %logistic loss
            H2 = Q;
            
            Q2 = Q < 37;
            
            H2(Q2) = log(1+exp(Q(Q2)));
        end
    end

    f = reg+c*sum(H2); %Value of the objective function
    
    count1 = count1+1;

    diff = fold-f; %Decrease in objective function

    % Check Armijo rule.
    if diff >= armijo*lambda*(dfdL(:)'*dfdL(:)+escale*dfde'*dfde)
        stepcount = stepcount+1;
        
        if learningcurve && (mod(stepcount,interval) == 0 || interval == 1)
            learningTime(stepcount/interval+1) = toc(t0);
            
            G = G-diag(inf(n,1));
            [G I] = sort(G,2);
            G = G(:,2:end);
            Ysorted = Y(I(:,2:end));
            
            Ykclass = zeros(n,n-1);

            for i = 1:n
                totals = cumsum(bsxfun(@eq,classes,Ysorted(i,:)),2);

                [maxes indexes] = max(totals,[],1);

                uniqueness = sum(bsxfun(@eq,totals,maxes),1);

                for j = 2:n-1
                    if uniqueness(j) > 1
                        indexes(j) = indexes(j-1);
                    end
                end

                Ykclass(i,:) = classes(indexes);
            end
            
            Yeclass = zeros(n,1);

            for i = 1:n
                j = find(G(i,:) <= ei(i),1,'last');

                if isempty(j)
                    Yeclass(i) = Ykclass(i,1);
                else
                    Yeclass(i) = Ykclass(i,j);
                end
            end

            trainCurve(stepcount/interval+1) = mean(Y ~= Yeclass);
            
            if numTest > 0
                if diagonal
                    G = bsxfun(@plus,sum(bsxfun(@times,Kxtest.^2,L'),2),sum(bsxfun(@times,L,Kx'.^2),1))-2*Kxtest*bsxfun(@times,L,Kx');
                else
                    Gxtest = zeros(numTest,1);

                    for i = 1:numTest
                        Gxtest(i) = Kxtest(i,:)*C*Kxtest(i,:)';
                    end

                    Gx = zeros(1,n);

                    for i = 1:n
                        Gx(i) = Kx(i,:)*C*Kx(i,:)';
                    end

                    G = bsxfun(@plus,Gxtest,Gx)-2*Kxtest*C*Kx';
                end

                [G I] = sort(G,2);
                Ysorted = Y(I);
                
                Ykclass = zeros(numTest,n);

                for i = 1:numTest
                    totals = cumsum(bsxfun(@eq,classes,Ysorted(i,:)),2);

                    [maxes indexes] = max(totals,[],1);

                    uniqueness = sum(bsxfun(@eq,totals,maxes),1);

                    for j = 2:n
                        if uniqueness(j) > 1
                            indexes(j) = indexes(j-1);
                        end
                    end

                    Ykclass(i,:) = classes(indexes);
                end
                
                if emode == 0
                    ej = repmat(e,numTest,1);
                elseif emode == 1
                    ej = e(I(:,1));
                else
                    ej = [Kxtest ones(numTest,1)]*e;
                end
                
                Yeclass = zeros(numTest,1);

                for j = 1:numTest
                    i = find(G(j,:) <= ej(j),1,'last');

                    if isempty(i)
                        Yeclass(j) = Ykclass(j,1);
                    else
                        Yeclass(j) = Ykclass(j,i);
                    end
                end
                
                testCurve(stepcount/interval+1) = mean(Ytest ~= Yeclass);
            end
        end
        
        F(stepcount+1) = f;
        
        if backcount == 0;
            lambdaf = lambda;
        end
        
        if stepcount >= frame;
            sdiff = log(F(stepcount+1-frame)/f);
        else
            sdiff = inf;
        end
        
        if figures
            if diagonal
                [J I] = sort(abs(L),'descend');
                
                Xp = bsxfun(@times,J([1 2]),Kx(:,I([1 2]))');
            else
                [V J] = eig(C);
                [J I] = sort(abs(diag(J)),'descend');
                V = V(:,I);
                U = real(diag(sqrt(J(1:2)))*V(:,1:2)');
                U = [sign(U(1,1))*U(1,:); sign(U(2,1))*U(2,:)];

                Xp = U*Kx';
            end

            YY = Y+4*(H2 > 0);

            plot(Xp(1,Y == 1),Xp(2,Y == 1),'bs',Xp(1,Y == 2),Xp(2,Y == 2),'go',...
                 Xp(1,Y == 3),Xp(2,Y == 3),'rd',Xp(1,Y == 4),Xp(2,Y == 4),'cd')
            hold on
            plot(Xp(1,YY == 5),Xp(2,YY == 5),'bs','MarkerFaceColor','b')
            plot(Xp(1,YY == 6),Xp(2,YY == 6),'go','MarkerFaceColor','g')
            plot(Xp(1,YY == 7),Xp(2,YY == 7),'rd','MarkerFaceColor','r')
            plot(Xp(1,YY == 8),Xp(2,YY == 8),'cd','MarkerFaceColor','c')
            hold off
            title([sprintf('t = %g, f_t = %f, ',stepcount,f) '\epsilon_t' sprintf(' = %f',mean(ei))])
        end

        % Check if stopping criteria is met.
        if (sdiff <= convthresh) || stepcount >= maxsteps || toc(t0) >= maxtime
            if diagnostics
                i = stepcount+1;

                Time(i) = toc(t0);
                Reg(i) = reg;
                E(i) = mean(ei);
                Lambda(i) = lambda;
                Backcount(i) = backcount;
            end
            
            done = 1;
        else
            % Save values for next step.
            Lold = L;
            eold = e;
            fold = f;

            % Calculate dfdL.
            
            if loss1 == 1 %hinge loss
                H1p = O > 0; %Gradient of inner hinge function for each point pair
            elseif loss1 == 2 %smooth hinge loss
                H1p = O;
                H1p(O1) = 1;
            elseif loss1 == 3 %quadratic loss
                H1p = 2.*O;
            else %logistic loss
                H1p = ones(n);
                
                P = exp(O(O2));
                H1p(O2) = P./(1+P);
            end

            if s ~= 1
                H1p = S.*H1p;
            end

            if loss2 == 0
                H2p = On;
            else
                if loss2 == 1 %hinge loss
                    H2p = Q > 0; %Gradient of outer hinge function for each point
                elseif loss2 == 2 %smooth hinge loss
                    H2p = Q;
                    H2p(Q1) = 1;
                elseif loss2 == 3 %quadratic loss
                    H2p = 2.*Q;
                else %logistic loss
                    H2p = On;
                    
                    R = exp(Q(Q2));
                    H2p(Q2) = R./(1+R);
                end
            end

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            Grad = bsxfun(@times,T,H2p).*H1p; %Gradient of active point pairs

            Coeff = Grad+Grad';
            Coeff = diag(sum(Coeff))-Coeff; %Coefficients of outer product pairs
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            if diagonal
                dfdLs = c*sum(Kx'.*(Coeff*Kx)',2);
            else
                dfdLs = c*Kx'*Coeff*Kx; %Gradient of slack term
            end
            
            if regularization == 1 %1-norm regularization
                if nonlinear
                    if diagonal
                        dfdLr = diag(Kz);
                    else
                        dfdLr = Kz; %Gradient of regularization term
                    end
                else
                    if diagonal
                        dfdLr = Or;
                    else
                        dfdLr = Ir;
                    end
                end
            elseif regularization == 2 %2-norm regularization
                if nonlinear
                    if diagonal
                        dfdLr = sum(Kz.*bsxfun(@times,L,Kz)',2);
                    else
                        dfdLr = Kz*C*Kz; %Gradient of regularization term
                    end
                else
                    if diagonal
                        dfdLr = L;
                    else
                        dfdLr = C;
                    end
                end
            else %no regularization
                dfdLr = 0;
            end
            
            if convex
                dfdL = dfdLr+dfdLs;
            else
                dfdL = 2*L*(dfdLr+dfdLs);
            end
            
            count2 = count2+1;

            % Calculate dfde.
            if emode == 0
                dfde = -c*sum(sum(Grad)); %Gradient of the objective function with respect to e
            elseif emode == 1
                dfde = -c*sum(Grad,2);
            elseif emode == 2
                dfde = -c*[Kx On]'*sum(Grad,2);
            end

            if diagnostics
                i = stepcount+1;

                Time(i) = toc(t0);
                Numactive1(i) = sum(sum(abs(Grad),2) ~= 0);
                Numactive2(i) = sum(sum(abs(Grad) ~= 0));
                Numactive3(i) = sum(sum((abs(Grad)+abs(Grad)').*(ones(n)-eye(n))) ~= 0);
                Reg(i) = reg;
                E(i) = mean(ei);
                Lambda(i) = lambda;
                Backcount(i) = backcount;
                DfdLnorm(i) = norm(dfdL,'fro');
                Dfde(i) = norm(dfde);
            end

            % Increase lambda.
            lambda = lambdaup*lambda;

            % Reset backcount.
            backcount = 0;

            % Update L and e.
            L = L-lambda*dfdL;
            e = e-lambda*escale*dfde;
        end
    else
        % Check if stopping criteria is met.
        if backcount >= maxbackcount
            % Revert to previous values.
            L = Lold;
            e = eold;
            f = fold;

            done = 1;
        else
            % Decrease learning rate by half.            
            lambda = lambdadown*lambda;

            % Increment backcount.
            backcount = backcount+1;
            
            % Update L and e with smaller learning rate.
            L = Lold-lambda*dfdL;
            e = eold-lambda*escale*dfde;
        end
    end

    % Check if done.
    if ~done
        if convex && metric
            % Project L onto the PSD cone.
            
            if diagonal
                L = max(L,0);
            else
                [V J] = eig(L);
                
                J = diag(J);
                J = max(0,~imag(J).*J);

                L = V*diag(J)*V';
            end
            
            count3 = count3+1;
        end
        
        if zerosum && nonlinear
            if convex
                L = Hr*L*Hr;
            else
                L = L*Hr;
            end
            
            count4 = count4+1;
        end

        if ((emode == 0 || emode == 1) && (~convex || (convex && metric))) || (nonlinear && metric)
            if ~convex || (convex && metric)
                % Project e onto the set of positive real values.
                e = max(0,e);
            end
        end
    else
        % Display which stopping criteria was met.
%         if stepcount >= maxsteps
%             display('Reached maximum number of steps.')
%         elseif backcount >= maxbackcount
%             display('Reached maximum number of backtracks.')
%         elseif sdiff <= convthresh
%             display('Reached convergence threshold.')
%         elseif toc(t0) >= maxtime
%             display('Reached maximum runtime.')
%         end
    end

    if figures
        drawnow
    end
end

runtime = toc(t0);

%% Display Results

% Calculate LOOE with KNN using the learned distance. Diplay data along with
% incorrectly classifide data points (solid markers) and reference points (+).

if diagonal
    C = diag(L);
elseif convex
    C = L;
else
    C = L'*L;
end

% figure
% imshow(-C,[min(min(-C)) max(max(-C))]);
% title('C')

A = Kx*C*Kx'; %Weighted inner product matrix

G = bsxfun(@plus,diag(A),diag(A)')-2*A;

[G I] = sort(G,2);
B = Y(I(:,2:k+1));
[M R] = mode(B,2);
M(R == 1) = B(R == 1,1);
W = Y ~= M;
YY = Y+4*W;
LOOE2 = sum(W)/n;

% Calculate test error with KNN using the learned distance.
if numTest > 0
    Gxtest = zeros(numTest,1);
    
    for i = 1:numTest
        Gxtest(i) = Kxtest(i,:)*C*Kxtest(i,:)';
    end
    
    Gx = zeros(1,n);
    
    for i = 1:n
        Gx(i) = Kx(i,:)*C*Kx(i,:)';
    end
    
    G = bsxfun(@plus,Gxtest,Gx)-2*Kxtest*C*Kx';
    
    [G I] = sort(G,2);
    B = Y(I(:,1:k));
    [M R] = mode(B,2);
    M(R == 1) = B(R == 1,1);
    W = Ytest ~= M;
    TEST2 = sum(W)/numTest;
end

[V J] = eig(C);
J = diag(J);
[dummy I] = sort(abs(J),'descend');
J = J(I);
V = V(:,I);
if (convex && r > 2) || (~convex && d > 2)
    U = real(diag(sqrt(J(1:3)))*V(:,1:3)');
else
    U = real(diag(sqrt(J(1:2)))*V(:,1:2)');
end

Xp = U*Kx';
Zp = U*Kz';

figure
plot(Xp(1,Y == 1),Xp(2,Y == 1),'bs',Xp(1,Y == 2),Xp(2,Y == 2),'go',...
     Xp(1,Y == 3),Xp(2,Y == 3),'rd',Xp(1,Y == 4),Xp(2,Y == 4),'cd')
hold on
plot(Xp(1,YY == 5),Xp(2,YY == 5),'bs','MarkerFaceColor','b')
plot(Xp(1,YY == 6),Xp(2,YY == 6),'go','MarkerFaceColor','g')
plot(Xp(1,YY == 7),Xp(2,YY == 7),'rd','MarkerFaceColor','r')
plot(Xp(1,YY == 8),Xp(2,YY == 8),'cd','MarkerFaceColor','c')
if nonlinear && reference
    plot(Zp(1,:),Zp(2,:),'+k')
end
hold off
title(sprintf('Learned Metric, LOOE = %g%%',LOOE2*100))
axis equal

% if (convex && r > 2) || (~convex && d > 2)
%     figure
%     plot3(Xp(1,YY == 1),Xp(2,YY == 1),Xp(3,YY == 1),'bs',Xp(1,YY == 2),Xp(2,YY == 2),Xp(3,YY == 2),'go',...
%           Xp(1,YY == 3),Xp(2,YY == 3),Xp(3,YY == 3),'rd',Xp(1,YY == 4),Xp(2,YY == 4),Xp(3,YY == 4),'cd')
%     hold on
%     plot3(Xp(1,YY == 5),Xp(2,YY == 5),Xp(3,YY == 5),'bs','MarkerFaceColor','b')
%     plot3(Xp(1,YY == 6),Xp(2,YY == 6),Xp(3,YY == 6),'go','MarkerFaceColor','g')
%     plot3(Xp(1,YY == 7),Xp(2,YY == 7),Xp(3,YY == 7),'rd','MarkerFaceColor','r')
%     plot3(Xp(1,YY == 8),Xp(2,YY == 8),Xp(3,YY == 8),'cd','MarkerFaceColor','c')
%     hold off
%     title(sprintf('Learned Metric, LOOE = %g%%',LOOE2*100))
%     axis equal
% end

if emode == 0
    ei = e*ones(n,1);
elseif emode == 1
    ei = e;
elseif emode == 2
    ei = [Kx ones(n,1)]*e;
end

J = sort(real(J),'descend');

% figure
% if emode == 0
%     stem(J'/sum(abs(J)));
%     axis([0 r+1 min(min(J'/sum(abs(J))),0) max(J'/sum(abs(J)))])
%     title('Eignenvalues')
% else
%     subplot(2,1,1)
%     stem(J'/sum(abs(abs(J))));
%     axis([0 r+1 min(min(J'/sum(abs(J))),0) max(J'/sum(abs(J)))])
%     title('Eignenvalues')
%     subplot(2,1,2)
%     stem(ei)
%     axis([0 n+1 min([ei; 0]) max(ei)+eps])
%     title('\epsilon_i')
% end

if learningcurve
    trainCurve = trainCurve(1:floor(stepcount/interval)+1);
    testCurve = testCurve(1:floor(stepcount/interval)+1);
    learningTime = learningTime(1:floor(stepcount/interval)+1);
    
    figure
    subplot(2,1,1)
    hold on
    plot((0:floor(stepcount/interval))*interval,trainCurve*100,'r--')
    if numTest > 0
        plot((0:floor(stepcount/interval))*interval,testCurve*100,'b:')
        legend('LOOE','Test Error')
    end
    axis([0 stepcount 0 max([trainCurve testCurve])*100+eps])
    title('Learning Curves')
    ylabel('Error (%)')
    xlabel('t')
    subplot(2,1,2)
    hold on
    plot(learningTime,trainCurve*100,'r--')
    if numTest > 0
        plot(learningTime,testCurve*100,'b:')
        legend('LOOE','Test Error')
    end
    axis([0 runtime 0 max([trainCurve testCurve])*100+eps])
    title('Learning Curves')
    ylabel('Error (%)')
    xlabel('s')
end
    
%% Display Diagnostic Plots

if diagnostics
    Time = Time(1:stepcount+1);
    Numactive1 = Numactive1(1:stepcount);
    Numactive2 = Numactive2(1:stepcount);
    Numactive3 = Numactive3(1:stepcount);
    F = F(1:stepcount+1);
    Reg = Reg(1:stepcount+1);
    E = E(1:stepcount+1);
    Lambda = Lambda(2:stepcount+1);
    Backcount = Backcount(2:stepcount+1);
    DfdLnorm = DfdLnorm(1:stepcount);
    Dfde = Dfde(1:stepcount);

    figure
    subplot(2,1,1)
    hold on
    if regularization
        plot(0:stepcount,log10(F-Reg),'r--')
        plot(0:stepcount,log10(Reg),'b:')
    end
    plot(0:stepcount,log10(F),'g-.')
    axis([0 stepcount log10(min([Reg F-Reg])) log10(max(F))+eps])
    if regularization
        legend('Slack','Regularization','Total')
    end
    title('Value of Objective Function')
    ylabel('log_{10}(f_t)')
    xlabel('t')
    subplot(2,1,2)
    hold on
    if regularization
        plot([0 Time(1:end-1)],log10(F-Reg),'r--')
        plot([0 Time(1:end-1)],log10(Reg),'b:')
    end
    plot([0 Time(1:end-1)],log10(F),'g-.')
    axis([0 Time(end-1) log10(min([Reg F-Reg])) log10(max(F))+eps])
    if regularization
        legend('Slack','Regularization','Total')
    end
    title('Value of Objective Function')
    ylabel('log_{10}(f_t)')
    xlabel('s')
        
    figure
    subplot(2,1,1)
    hold on
    plot(1:stepcount,log10(F(1:end-1)-F(2:end)),'b-')
    plot(frame:stepcount,log10(F(1:end-frame)-F(frame+1:end)),'g--')
    axis([0 stepcount log10(min([F(1:end-1)-F(2:end) F(1:end-frame)-F(frame+1:end)])) log10(max([F(1:end-1)-F(2:end) F(1:end-frame)-F(frame+1:end)]))+eps])
    legend('\delta = 1',['\delta = ' sprintf('%g',frame)])
    title('Difference from Previous Iteration')
    ylabel('log_{10}(f_t-f_{t-\delta})')
    subplot(2,1,2)
    hold on
    plot(1:stepcount,log10(log(F(1:end-1)./F(2:end))),'b-')
    plot(frame:stepcount,log10(log(F(1:end-frame)./F(frame+1:end))),'g--')
    plot([0 stepcount],[log10(convthresh) log10(convthresh)],'k:')
    axis([0 stepcount log10(min([log(F(1:end-1)./F(2:end)) log(F(1:end-frame)./F(frame+1:end)) convthresh])) log10(max([log(F(1:end-1)./F(2:end)) log(F(1:end-frame)./F(frame+1:end))]))+eps])
    legend('\delta = 1',['\delta = ' sprintf('%g',frame)],'threshold')
    title('Scaled Difference from Previous Iteration')
    ylabel('log_{10}(ln(f_{t-\delta}/f_t))')
    xlabel('t')
    
    figure
    subplot(3,1,1)
    plot(0:stepcount-1,Numactive1/n*100)
    axis([0 stepcount 0 100])
    title('Percentage of Active Outer Loss Functions')
    ylabel('%')
    subplot(3,1,2)
    plot(0:stepcount-1,Numactive2/n^2*100)
    axis([0 stepcount 0 100])
    title('Percentage of Active Inner Loss Functions')
    ylabel('%')
    subplot(3,1,3)
    plot(0:stepcount-1,Numactive3/n*100)
    axis([0 stepcount 0 100])
    title('Percentage of Active Points')
    ylabel('%')
    xlabel('t')
    
    figure
    subplot(3,1,1)
    plot(0:stepcount,Time-[0 Time(1:end-1)])
    axis([0 stepcount 0 max(Time-[0 Time(1:end-1)])+eps])
    title('Time')
    ylabel('Seconds')
    subplot(3,1,2)
    stem(1:stepcount,Backcount)
    axis([0 stepcount 0 max(Backcount)+1])
    title('Number of Consecutive Backtracks')
    ylabel('Count')
    subplot(3,1,3)
    plot(1:stepcount,log10(Lambda))
    axis([0 stepcount log10(min(Lambda)) log10(max(Lambda))+eps])
    title('Learning Rate')
    ylabel('log_{10}(\lambda_t)')
    xlabel('t')
    
    figure
    subplot(3,1,1)
    plot(0:stepcount-1,log10(DfdLnorm))
    axis([0 stepcount min(log10(DfdLnorm)) max(log10(DfdLnorm))+eps])
    title('Magnitude of Gradient with Respect to L')
    ylabel('log_{10}(||df/dL||_F)')
    subplot(3,1,2)
    plot(0:stepcount-1,log10(abs(Dfde)))
    axis([0 stepcount log10(min(abs(Dfde))) log10(max(abs(Dfde)))+eps])
    title('Magnitude of Gradient with Respect to e')
    ylabel('log_{10}(||df/de||)')
    subplot(3,1,3)
    plot(0:stepcount,E)
    axis([0 stepcount 0 max(E)+eps])
    title('Mean Kernel Bandwidth')
    ylabel('\epsilon_t')
    xlabel('t')
end

%% Dispay final value of some variables.

% display(sprintf('Runtime: %g seconds',runtime))
% display(sprintf('Stepcount: %g',stepcount))
% display(sprintf('Backcount: %g',backcount))
% display(sprintf('Scaled difference: %g',sdiff))
% display(sprintf('Objective function value: %g',f))
% display(sprintf('Learning rate: %g',lambdaf))
display(sprintf('LOOE before training: %g%%',LOOE1*100))
display(sprintf('LOOE after training: %g%%',LOOE2*100))
if numTest > 0
    display(sprintf('Test error before training: %g%%',TEST1*100))
    display(sprintf('Test error after training: %g%%',TEST2*100))
end
% display(sprintf('Mean epsilon: %g%',mean(ei)))
% display(sprintf('Steps per second: %g',stepcount/runtime))
% display(sprintf('Objective calculations per second: %g',count1/runtime))
% display(sprintf('Number of times f is computed: %g',count1))
% display(sprintf('Number of times df is computed: %g',count2))
% display(sprintf('Number of psd projections: %g',count3))
% display(sprintf('Number of zerosum prejections: %g',count4))
% display(sprintf('Trace of C: %g',trace(C)))
% display(sprintf('Sum of C: %g',sum(sum((C)))))
% display(sprintf('Max row sum of C: %g',max(abs(sum((C))))))
% display(sprintf('Rank of C: %g\n',sum(cumsum(J/sum(J)) < .99)))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

emin = 0;
emax = 2*mean(ei);
numE = 100;

E = [linspace(emin,emax,numE) mean(ei)];
Fe = zeros(1,numE+1);

% Calculate regularization term.
if regularization == 1 %1-norm regularization
    if nonlinear
        reg = C(:)'*Kz(:); %Value of the regularization term
    else
        reg = trace(C);
    end
elseif regularization == 2 %2-norm regularization
    if nonlinear
        Temp1 = C*Kz;
        Temp2 = Temp1';
        reg = .5*Temp1(:)'*Temp2(:);
    else
        reg = .5*C(:)'*C(:);
    end
else %no regularization
    reg = 0;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

A = Kx*C*Kx'; %Weighted inner product matrix

G = bsxfun(@plus,diag(A),diag(A)')-2*A;

for i = 1:numE+1
    if loss1 < 4
        O = max(1+T.*bsxfun(@minus,G,E(i)),0); %Argument of inner hinge function for each point pair
    else
        O = T.*bsxfun(@minus,G,E(i));
    end
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Calculate slack penalty.
    if loss1 == 1 %hinge loss
        H1 = O; %Value of inner hinge function for each point pair
    elseif loss1 == 2 %smooth hinge loss
        H1 = .5.*O.^2;
        
        O1 = O > 1;
        
        H1(O1) = O(O1)-.5;
    elseif loss1 == 3 %quadratic loss
        H1 = O.^2;
    else %logistic loss
        H1 = log(1+exp(O));
    end
    
    if loss2 == 0
        H2 = sum(H1,2);
    else
        if loss2 < 4
            Q = max(1+sum(H1,2)-N,0);
        else
            Q = sum(H1,2)-N;
        end

        if loss2 == 1 %hinge loss
            H2 = Q; %Value of outer hinge function for each point
        elseif loss2 == 2 %smooth hinge loss
            H2 = .5.*Q.^2;

            Q1 = Q > 1;

            H2(Q1) = Q(Q1)-.5;
        elseif loss2 == 3 %quadratic loss
            H2 = Q.^2;
        else %logistic loss
            H2 = log(1+exp(Q));
        end
    end

    Fe(i) = reg+c*sum(H2); %Value of the objective function
end