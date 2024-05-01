function anchor = Kmeanspp(X,m)

% ======================    Input    ====================== %
% ------     X    : sample matrix                ||  d * n  %
% ------     m    : the number of anchors        ||  1 * 1  %
% ========================================================= %

% ======================   Output    ====================== %
% ------ anchor   : the data matrix of anchors              %
% ========================================================= %
L = [];
L1 = 0;

while length(unique(L)) ~= m
    
    % ===========  Initialization of K-means++ ============ %
    anchor = X(:,1 + round(rand*(size(X, 2)-1)));  % select samples as initial anchor randomly
    L = ones(1, size(X,2));
    for i = 2:m
        D = X - anchor(:, L);
        D = sqrt(dot(D, D, 1));
        [~, index] = max(D);
        anchor(:, i) = X(:, index);                % choose nearset point from existing samples as newest anchor
        [~,L] = max(bsxfun(@minus, 2 * real(anchor' * X), dot(anchor,anchor,1).')); 
    end
    
    % ===============  Updating of K-means++  ============ %
    while any(L ~= L1)
        L1 = L;
        for i = 1:m 
            l = L==i; 
            anchor(:,i) = sum(X(:,l),2)/sum(l); 
        end
        [~,L] = max(bsxfun(@minus,2*real(anchor'*X),dot(anchor,anchor,1).'),[],1);
    end    
    
end