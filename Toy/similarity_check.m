function localStruct = similarity_check(B, F, G)

% ======================    Input    ====================== %
% ------     B    : anchor graph matrix          ||  n * m  %
% ------     F    : label matrix of samples      ||  n * c  %
% ------     G    : label matrix of anchors      ||  m * c  %
% ========================================================= %

[n, m] = size(B);
I_nm = ones(n, m);
B_nm = B;
B_nm(B_nm ~= 0) = 1;

allDist   = trace( F' * diag(sum(I_nm, 2)) * F - 2 * F' * I_nm * G + G' * diag(sum(I_nm, 1)) * G );
localDist = trace( F' * diag(sum(B_nm, 2)) * F - 2 * F' * B_nm * G + G' * diag(sum(B_nm, 1)) * G );

localStruct = localDist/allDist;

end
