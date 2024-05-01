function [new_F, new_G, obj]=update_Y(X, anchors, B, Label, F, G, d, p)

% ======================    Input    ====================== %
% ------     X    : sample matrix                ||  d * n  %
% ------  anchors : anchor matrix                ||  d * m  %
% ------   Label  : label matrix                 ||  1 * n  %
% ------     B    : anchor graph matrix          ||  n * m  %
% ------     d    : the number of labels = c     ||  1 * 1  %
% ------     F    : label matrix of samples      ||  n * c  %
% ------     G    : label matrix of anchors      ||  m * c  %
% ------     p    : regularization parameter     ||  1 * 1  %
% ========================================================= %

% ======================   Output    ====================== %
% ------ new_F    : label matrix of samples after updating  %
% ------ new_G    : label matrix of anchors after updating  %
% ------  obj     : the value of objective function         %
% ========================================================= %

% ======================     Ref     ====================== %
% Jingyu Wang, Shengzhao Guo and Feiping Nie and Xuelong Li %
%      Local-Global Fuzzy Clustering with Anchor Graph      %
%         IEEE Transactions on Fuzzy Systems, 2024.         %
% ==============  ( code by Shengzhao Guo ) =============== %

% =====================  Pre-setting  ===================== %
numIteration = zeros(1,5);              % The number of itrations
valueObj     = zeros(1,5);              % The value of Objective function
valueAcc     = zeros(1,5);              % The value of Accuracy ( Acc )
valueNMI     = zeros(1,5);              % The value of Normalized Mutual Information ( NMI )
localStruct  = zeros(1,5);              % The local structure of latent fuzzy feature Y
globalStruct = zeros(1,5);              % The global structure of latent fuzzy feature Y

obj      = obj_cal(X,anchors,F,G,B,p);  % The initial value of objective function
error    = 1;                           % The initial value dropped after updating
countNum = 1;                           % The initialization of iteration number
new_F    = F;
new_G    = G;

% =====================  Optimization  ==================== %
while error > 10^-4
    
    % Assign labels according to maximum membership
    [~, index] = max(new_F');
    [Acc, NMI, ~] = ClusteringMeasure(Label, index);
    
    valueAcc(countNum)     = Acc;
    valueNMI(countNum)     = NMI;
    valueObj(countNum)     = obj;
    numIteration(countNum) = countNum;
    
    disp(['|| In the ',    num2str(countNum, '%02d'),     'th iteration ',...
          '>>>> The value objective function :: ', num2str(obj,'%.4f')]);
    
    % Update fuzzy membership matrix of samples
    new_F = update_F(X, anchors, B, new_F, new_G, d, p);
    
    % Update fuzzy membership matrix of anchors
    new_G = update_G(X, anchors, B, new_F, new_G, d, p);

    % if countNum == 1
    %     viewFMember(new_F, 2);           % The visualization of fuzzy membership matrix F after one iteration
    % end
    
    obj_new = obj_cal(X, anchors, new_F, new_G, B, p);
    error = abs(obj - obj_new);
    
    obj = obj_new; countNum = countNum + 1;

    if countNum > 50
        break
    end
    
end

yyaxis left;
plot(numIteration,valueAcc*100,'o-b',numIteration,valueNMI*100,'s-b');
xlabel('Number of Iterations');
ylabel('Clustering Performance (%)');
legend('ACC','NMI','Location','east');
set(gca,'ycolor','b'); 
set(gca,'ylim',[0,100],'yTick',[0:10:100])
set(gca,'xlim',[0,20],'xTick',[0:5:20])
grid on
grid minor

yyaxis right;
plot(numIteration,valueObj,'*-r','HandleVisibility','off');
ylabel('Objective Function Value');
set(gca,'xlim',[0,50],'xTick',[0:10:50])
set(gca,'Units','inches','OuterPosition',[0 0 5.83 4.37])

