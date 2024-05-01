function value = obj_cal(X, anchors, F, G, B, p)

% ======================    Input    ====================== %
% ------     X    : sample matrix                ||  d * n  %
% ------  anchors : anchor matrix                ||  d * m  %
% ------     B    : anchor graph matrix          ||  n * m  %
% ------     F    : label matrix of samples      ||  n * c  %
% ------     G    : label matrix of anchors      ||  m * c  %
% ------     p    : regularization parameter     ||  1 * 1  %
% ========================================================= %

m=size(anchors,2);
n=size(X,2);

T = - 1 / (n + m) * ones(n, m);
H_n = eye(n, n) - 1 / (n + m) * ones(n, n);
H_m = eye(m, m) - 1 / (n + m) * ones(m, m);

value = 2 * ( trace(F' * ( diag(sum(B, 2)) - p * H_n) * F) + trace(G' * ( diag(sum(B, 1)) - p * H_m) * G) - 2 * trace(F' * (B + p * T) * G));