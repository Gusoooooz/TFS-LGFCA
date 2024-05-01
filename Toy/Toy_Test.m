clc; close all

dataName = 'spheres';
load(['..\datasets\', dataName, '.mat']);
X = X';                              % data matrix of sample  | d * n
Label = label;                       % label vector of sample | n * 1
numLabel = length(unique(Label));    % number of clusters = c
numSample = size(X, 2);

disp(['|| ================ Toy Sample Experiment >>> ', dataName, ' ================']);

% =====================  Para-setting  ===================== %
K_nearest = 2;     % the number of nearset anchors for each sample
numAnchor = 8;     % the number of anchors in anchor graph
lambda = 0.2;      % the value of regularization parameter

% ====================  Initialization  ==================== %
anchors = Kmeanspp(X, numAnchor);            % Generation of anchors using K-means++
B = ConstructA_NP(X, anchors, K_nearest);    % Construction of sparse anchor graph B
Y=rand(numSample + numAnchor, numLabel);     % Random initialization of Y
Y = bsxfun(@rdivide,Y,sum(Y,2));             % Normalization of Y >>> Y >= 0 & Y^T * 1 = 1
F = Y(1:numSample, :);
G = Y(numSample + 1:numSample + numAnchor,:);

% =====================  Optimization  ===================== %
[F, G, obj] = update_Y(X, anchors, B, Label, F, G, numLabel, lambda);

% ======================  Measurement  ===================== %
[~, L] = max(F'); [acc, nmi, purity] = ClusteringMeasure(Label, L); % Assign labels according to maximum membership

disp(['||   ACC >>> ', num2str(acc * 100, '%.2f'), '   ||   NMI >>> ', num2str(nmi * 100, '%.2f'), '   ||   Purity >>> ', num2str(purity * 100, '%.2f'), '   ||']);