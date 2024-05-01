
function viewFMember(S, k)

% ======================    Input    ====================== %
% ------     S    : fuzzy membership matrix      ||  d * n  %
% ------     k    : 1 >> non-zero distribution  2 >> hotmap %
% ========================================================= %

S=S';
[m, n] = size(S);

if k==1
    for j=1:n
        for i=1:m
            if S(i,j)~=0
                S(i,j)=1; % the visualization of non-zero elements
            end
        end
    end
else
    for j=1:n
        for i=1:m
            a=S(i,j);    % the visualization of hotmap
            if a<1
                S(i,j)=a;
            else
                S(i,j)=1;
            end
        end
    end
end

     
figure

o=heatmap(S,'YLabel', 'Soft label for each clusters', 'XLabel', 'Data point ID','CellLabelColor' , 'none','GridVisible','off');
o.XDisplayLabels = nan(size(o.XDisplayData));

annotation('textbox',...
    [0.115285714285714 0.916666667872952 0.0535714275070599 0.0642857130794299],...
    'String',{'1'},...
    'LineStyle','none');
annotation('textbox',...
    [0.449214285714283 0.914285715492001 0.0803571409944978 0.0642857130794299],...
    'String',{'100'},...
    'LineStyle','none');
annotation('textbox',...
    [0.79742857142857 0.914285715492001 0.0803571409944978 0.0642857130794299],...
    'String',{'200'},...
    'LineStyle','none');

