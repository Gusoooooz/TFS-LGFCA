function A = ConstructA_NP(TrainData, Anchor,k)

% ======================    Input    ====================== %
% ------ TrainData : sample matrix               ||  d * n  %
% ------     m     : the number of anchors       ||  1 * 1  %
% ========================================================= %

% ======================   Output    ====================== %
% ------ anchor   : the data matrix of anchors              %
% ========================================================= %

Dis = EuDist2(TrainData',Anchor',0);

[~,idx] = sort(Dis,2); 
idx1 = idx(:,1:k+1);
clear idx;
[~,anchor_num] = size(Anchor);
[~,num] = size(TrainData);
A = zeros(num,anchor_num);

for i = 1:num
    id = idx1(i,1:k+1);
    di = Dis(i,id);
    A(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);
end;

A = sparse(A);
