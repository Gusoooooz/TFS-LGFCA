function divergence = divergence_check(F, G)

% ======================    Input    ====================== %
% ------     F    : label matrix of samples      ||  n * c  %
% ------     G    : label matrix of anchors      ||  m * c  %
% ========================================================= %

n = size(F,1);
m = size(G,1);
T = - 1 / (n + m) * ones(n, m);
H_n = eye(n, n) - 1 / (n + m) * ones(n, n);
H_m = eye(m, m) - 1 / (n + m) * ones(m, m);

divergence = 2 * ( trace(F' *  H_n  * F) + trace(G' * H_m * G) + 2 * trace(F' * T * G));

end