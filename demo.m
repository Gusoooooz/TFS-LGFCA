clc; close all

dataName = 'USPS';
load(['.\datasets\', dataName, '.mat']);
X = X';            % data matrix of sample  | d * n
Label = label;     % label vector of sample | n * 1
numLabel = length(unique(Label));    % number of clusters = c
numSample = size(X, 2);              % number of samples 

disp(['|| ================ Real-world Data Experiment >>> ', dataName, ' ================']);

% The best para-setting for USPS  :  numAnchor = 1500 | lambda = 8
% The best para-setting for MNIST : numAnhcor = 1500 | lambda = 18
% =====================  Para-setting  ===================== %
K_nearest = 72;    % the number of nearset anchors for each sample
numAnchor = 1500;  % the number of anchors in anchor graph
lambda = 8;        % the value of regularization parameter

% ====================  Initialization  ==================== %
tic;
anchors = Kmeanspp(X, numAnchor);            % Generation of anchors using K-means++
timeGen = toc;
disp(['|| ===== The generation of anchors is completed in ', num2str(timeGen, '%.4f'), ' seconds ===== ']);

tic;
B = ConstructA_NP(X, anchors, K_nearest);    % Construction of sparse anchor graph B
timeGraph = toc;
disp(['|| == The construction of anchor graph is completed in ', num2str(timeGraph, '%.4f'), ' seconds ==']);

tic
Y=rand(numSample + numAnchor, numLabel);     % Random initialization of Y
Y = bsxfun(@rdivide,Y,sum(Y,2));
F = Y(1:numSample, :);
G = Y(numSample + 1:numSample + numAnchor,:);
timeMatrix = toc;
disp(['|| = The Initialization of fuzzy matrix is completed in ', num2str(timeMatrix, '%.4f'),' seconds =']);

% =====================  Optimization  ===================== %
tic;
[F, G, obj] = update_Y(X, anchors, B, Label, F, G, numLabel, lambda);
timeUpdating = toc;
disp(['|| == The Optimization of fuzzy matrix is completed in ', num2str(timeUpdating, '%.4f'),' seconds ==']);

disp(['|| ======== The LGFCA program is completed in ', num2str(timeGen + timeGraph + timeMatrix + timeUpdating, '%.4f'),' seconds ==========']);

% ======================  Measurement  ===================== %
[~, L] = max(F'); [acc, nmi, purity] = ClusteringMeasure(Label, L); % Assign labels according to maximum membership

disp(['||    ACC >>> ', num2str(acc * 100,'%.2f'), '    ||    NMI >>> ', num2str(nmi * 100, '%.2f'), '    ||    Purity >>> ', num2str(purity * 100,'%.2f'), '   ||']);