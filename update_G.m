function G = update_G(X, anchor, B, F, G, d, p)

% ======================    Input    ====================== %
% ------     X    : sample matrix                ||  d * n  %
% ------  anchors : anchor matrix                ||  d * m  %
% ------     B    : anchor graph matrix          ||  n * m  %
% ------     d    : the number of labels = c     ||  1 * 1  %
% ------     F    : label matrix of samples      ||  n * c  %
% ------     G    : label matrix of anchors      ||  m * c  %
% ------     p    : regularization parameter     ||  1 * 1  %
% ========================================================= %

% ======================   Output    ====================== %
% ------ output_F : label matrix of samples after updating  %
% ========================================================= %

% ======================     Ref     ====================== %
% Jingyu Wang, Shengzhao Guo and Feiping Nie and Xuelong Li %
%      Local-Global Fuzzy Clustering with Anchor Graph      %
%         IEEE Transactions on Fuzzy Systems, 2024.         %
% ==============  ( code by Shengzhao Guo ) =============== %

n = size(X, 2);       % the number of samples
m = size(anchor, 2);  % the number of anchors

for i = 1 : m 
    
    v = (B(:, i)' - p / (n + m)) * F;
    v = v - p / (n + m) * ones(1, m) * G + p / (n + m) * G(i, :);

    temp = 1 / (sum(B(:, i)) - p + p / (n + m));
    v = temp * v;
    
    if temp > 0 
        [G(i, :), ~] = EProjSimplex_1(v, 1);
    else
        G(i, :) = zeros(1, d);
        [~, position] = sort(v, 'ascend');
        G(i, position(1,1)) = 1;
    end
end
