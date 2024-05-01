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
valueAcc     = zeros(1,5);              % The value of Accuracy
localStruct  = zeros(1,5);              % The local structure of latent fuzzy feature Y
globalStruct = zeros(1,5);              % The global structure of latent fuzzy feature Y

obj      = obj_cal(X,anchors,F,G,B,p);  % The initial value of objective function
error    = 1;                           % The initial value dropped after updating
countNum = 1;                           % The initialization of iteration number
new_F    = F;
new_G    = G;

viewFMember(F,2);                       % The visualization of initial fuzzy membership matrix F

% =====================  Optimization  ==================== %
while error > 10^-4
    
    % Assign labels according to maximum membership
    [~, index] = max(new_F');
    Acc = ClusteringMeasure(Label, index);
    
    valueAcc(countNum)     = Acc;
    valueObj(countNum)     = obj;
    numIteration(countNum) = countNum;
    globalStruct(countNum) = divergence_check(new_F, new_G);
    localStruct(countNum)  = similarity_check(B, new_F, new_G);
    
    disp(['|| In the ',    num2str(countNum, '%02d'),     'th iteration ',...
          '>>>> The value objective function :: ', num2str(obj,'%.4f')]);
    
    % Update fuzzy membership matrix of samples
    new_F = update_F(X, anchors, B, new_F, new_G, d, p);
    
    % Update fuzzy membership matrix of anchors
    new_G = update_G(X, anchors, B, new_F, new_G, d, p);

    % if countNum == 10
    %    viewFMember(new_F, 2);           % The visualization of fuzzy membership matrix F after one iteration
    % end
    
    obj_new = obj_cal(X, anchors, new_F, new_G, B, p);
    error = abs(obj - obj_new);
    
    obj = obj_new; countNum = countNum + 1;

    if countNum > 40
        break
    end
    
end

viewFMember(new_F,2);                    % The visualization of fuzzy membership matrix F obtained

% =====================  Visualization (Spheres)  ==================== %
%   Visualization of elements in fuzzy membership matrix F   %
figure
count_num=(1:1:300);
subplot(3,1,1);
plot(count_num,new_F(1:300,1),'Color',[0 .8 .2]);
set(gca,'YTick',[0:0.5:1]);
xticks([1 33 50 100 150 178 200 250 300])
xticklabels({'1','33','50','100','150','178','200','250','300'})
xlabel('Data point ID');
ylabel('Membership Value');
subplot(3,1,2);
plot(count_num,new_F(1:300,2),'Color',[0 .2 .8]);
set(gca,'YTick',[0:0.5:1]);
xticks([1 33 50 100 150 178 200 250 300])
xticklabels({'1','33','50','100','150','178','200','250','300'})
xlabel('Data point ID');
ylabel('Membership Value');
subplot(3,1,3);
plot(count_num,new_F(1:300,3),'Color',[0 .5 .7]);
set(gca,'YTick',[0:0.5:1]);
xticks([1 33 50 100 150 178 200 250 300])
xticklabels({'1','33','50','100','150','178','200','250','300'})
xlabel('Data point ID');
ylabel('Membership Value');

% Visualization of Local-Global structure during optimization %
figure;

ha(1) = axes('ycolor','r','yminortick','off','xminortick','on'); hold on; 
h(1) = line(numIteration - 1 ,localStruct*100, 'Marker','o','color','r','parent',ha(1),'linestyle','-');
ylim([0 100]);

grid on
grid minor
set(gca,'GridColor','k','MinorGridColor','k');
ax = gca;
ax.Layer = 'top';
xlim1 = get(ha(1),'xlim');

pos1=get(ha(1),'position');
ha(2) = axes('position',pos1,'color','none','ycolor',[0.8500 0.3250 0.0980],'yaxislocation','right','xlim',xlim1,  'xtick', []);
h(2) = line(numIteration - 1, globalStruct, 'color',[0.8500 0.3250 0.0980],'Marker','*','parent',ha(2));

pos1(1)=pos1(1)-0.02;
pos1(3) = pos1(3)*.86;
set([ha(1);ha(2)],'position',pos1);
pos3 = pos1;
pos3(3) = pos3(3)+.12;
xlim3 = xlim1;
xlim3(2) = xlim3(1)+(xlim1(2)-xlim1(1))/pos1(3)*pos3(3);
ha(3) = axes('position',pos3, 'color','none','ycolor','b','xlim',xlim3, 'xtick',[],'yaxislocation','right','yminortick','off');
h(3) = line(numIteration - 1, valueAcc*100,'color','b','Marker','s','parent',ha(3));

ylim3 = get(ha(3), 'ylim');
line([xlim1(2),xlim3(2)],[ylim3(1),ylim3(1)],'parent',ha(3),'color','w');

hylab = get([ha(1);ha(2);ha(3)],'ylabel');
hxlab = get(ha(1),'xlabel');
set(hylab{1},'string','Percentage of neighbouring distance (%)');
set(hylab{2},'string','Variance of latent representations');
set(hylab{3},'string','Accuracy (%)');
set(hxlab,'string', 'Number of Iterations');

