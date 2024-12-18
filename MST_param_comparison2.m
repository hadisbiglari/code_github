%% Wilcoxon signed-rank test
for i=1:size(Dist_AD,1)
    Distmax_AD(i) = max(Dist_AD(i,:));
    Distmax_HC(i) = max(Dist_HC(i,:));
    Distmax_MCI(i) = max(Dist_MCI(i,:));
end
% group1=[Distmax_AD Th_meanMatrix_AD   Diam_meanMatrix_AD  LF_meanMatrix_AD   BC_meanMatrix_AD'  Deg_meanMatrix_AD  k_AD  Ec_AD];
% group2=[Distmax_HC Th_meanMatrix_HC   Diam_meanMatrix_HC  LF_meanMatrix_HC   BC_meanMatrix_HC'  Deg_meanMatrix_HC  k_HC  Ec_HC];
% group3=[Distmax_MCI Th_meanMatrix_MCI   Diam_meanMatrix_MCI  LF_meanMatrix_MCI   BC_meanMatrix_MCI'  Deg_meanMatrix_MCI  k_MCI   Ec_MCI];
group1=[Th_meanMatrix_AD   Diam_meanMatrix_AD  LF_meanMatrix_AD   mean(BC_meanMatrix_AD)  mean(Deg_meanMatrix_AD)  k_AD  mean(Ec_AD)];
group2=[Th_meanMatrix_HC   Diam_meanMatrix_HC  LF_meanMatrix_HC   mean(BC_meanMatrix_HC)  mean(Deg_meanMatrix_HC)  k_HC  mean(Ec_HC)];
group3=[Th_meanMatrix_MCI   Diam_meanMatrix_MCI  LF_meanMatrix_MCI   mean(BC_meanMatrix_MCI)  mean(Deg_meanMatrix_MCI)  k_MCI   mean(Ec_MCI)];
for i = 1:size(BC_max_MSTmat_AD,1)
    group1(i)=[mean(cell2mat(BC_max_MSTmat_AD(i,:)))];
end
for i = 1:size(BC_max_MSTmat_HC,1)
    group2(i)=[mean(cell2mat(BC_max_MSTmat_HC(i,:)))];
end
for i = 1:size(BC_max_MSTmat_MCI,1)
    group3(i)=[mean(cell2mat(BC_max_MSTmat_MCI(i,:)))];
end

%%
group1=[Th_MSTmat_AD   Diam_MSTmat_AD  LF_MSTmat_AD   BC_max_MSTmat_AD    k_MSTmat_AD    Ec_MSTmat_AD_m    Deg_MSTmat_AD_m];
group2=[Th_MSTmat_HC   Diam_MSTmat_HC  LF_MSTmat_HC   BC_max_MSTmat_HC    k_MSTmat_HC  Ec_MSTmat_HC_m  Deg_MSTmat_HC_m];
group3=[Th_MSTmat_MCI   Diam_MSTmat_MCI  LF_MSTmat_MCI   BC_max_MSTmat_MCI    k_MSTmat_MCI   Ec_MSTmat_MCI_m  Deg_MSTmat_MCI_m];
%
group2 = [group2 cell(size(group2,1), size(group1,2)-size(group2,2))];
group3 = [group3 cell(size(group3,1), size(group1,2)-size(group3,2))];

%%
for i = 1:size(Th_MSTmat_AD, 1)
    for j = 1:size(Th_MSTmat_AD, 2)
        if isempty(Th_MSTmat_AD{i, j})
            Th_MSTmat_AD{i,j} = 0;
        end
    end
end
Th_MSTmat_AD = cell2mat(Th_MSTmat_AD);
Th_AD = [];
Th_AD = nonzeros(Th_MSTmat_AD);

for i = 1:size(Th_MSTmat_HC, 1)
    for j = 1:size(Th_MSTmat_HC, 2)
        if isempty(Th_MSTmat_HC{i, j})
            Th_MSTmat_HC{i,j} = 0;
        end
    end
end
Th_MSTmat_HC = cell2mat(Th_MSTmat_HC);
Th_HC = [];
Th_HC = nonzeros(Th_MSTmat_HC);

for i = 1:size(Th_MSTmat_MCI, 1)
    for j = 1:size(Th_MSTmat_MCI, 2)
        if isempty(Th_MSTmat_MCI{i, j})
            Th_MSTmat_MCI{i,j} = 0;
        end
    end
end
Th_MSTmat_MCI = cell2mat(Th_MSTmat_MCI);
Th_MCI = [];
Th_MCI = nonzeros(Th_MSTmat_MCI);

%%
for i = 1:size(Diam_MSTmat_AD, 1)
    for j = 1:size(Diam_MSTmat_AD, 2)
        if isempty(Diam_MSTmat_AD{i, j})
            Diam_MSTmat_AD{i,j} = 0;
        end
    end
end
Diam_MSTmat_AD = cell2mat(Diam_MSTmat_AD);
Diam_AD = [];
Diam_AD = nonzeros(Diam_MSTmat_AD);
% for i=1:size(Diam_MSTmat_AD,1)
%  Diam_AD(i) = mean(nonzeros(Diam_MSTmat_AD(i,:)));
% end

for i = 1:size(Diam_MSTmat_HC, 1)
    for j = 1:size(Diam_MSTmat_HC, 2)
        if isempty(Diam_MSTmat_HC{i, j})
            Diam_MSTmat_HC{i,j} = 0;
        end
    end
end
Diam_MSTmat_HC = cell2mat(Diam_MSTmat_HC);
Diam_HC = [];
Diam_HC = nonzeros(Diam_MSTmat_HC);
% for i=1:size(Diam_MSTmat_HC,1)
%  Diam_HC(i) = mean(nonzeros(Diam_MSTmat_HC(i,:)), 1)
% end

for i = 1:size(Diam_MSTmat_MCI, 1)
    for j = 1:size(Diam_MSTmat_MCI, 2)
        if isempty(Diam_MSTmat_MCI{i, j})
            Diam_MSTmat_MCI{i,j} = 0;
        end
    end
end
Diam_MSTmat_MCI = cell2mat(Diam_MSTmat_MCI);
Diam_MCI = [];
Diam_MCI = nonzeros(Diam_MSTmat_MCI);
% for i=1:size(Diam_MSTmat_MCI,1)
%  Diam_MCI(i) = mean(nonzeros(Diam_MSTmat_MCI(i,:)),1)
% end

%%
for i = 1:size(LF_MSTmat_AD, 1)
    for j = 1:size(LF_MSTmat_AD, 2)
        if isempty(LF_MSTmat_AD{i, j})
            LF_MSTmat_AD{i,j} = 0;
        end
    end
end
LF_MSTmat_AD = cell2mat(LF_MSTmat_AD);
LF_AD = [];
LF_AD = nonzeros(LF_MSTmat_AD);
% for i=1:size(LF_MSTmat_AD,1)
%  LF_AD(i) = mean(nonzeros(LF_MSTmat_AD(i,:)));
% end

for i = 1:size(LF_MSTmat_HC, 1)
    for j = 1:size(LF_MSTmat_HC, 2)
        if isempty(LF_MSTmat_HC{i, j})
            LF_MSTmat_HC{i,j} = 0;
        end
    end
end
LF_MSTmat_HC = cell2mat(LF_MSTmat_HC);
LF_HC = [];
LF_HC = nonzeros(LF_MSTmat_HC);
% for i=1:size(LF_MSTmat_HC,1)
%  LF_HC(i) = mean(nonzeros(LF_MSTmat_HC(i,:)), 1)
% end

for i = 1:size(LF_MSTmat_MCI, 1)
    for j = 1:size(LF_MSTmat_MCI, 2)
        if isempty(LF_MSTmat_MCI{i, j})
            LF_MSTmat_MCI{i,j} = 0;
        end
    end
end
LF_MSTmat_MCI = cell2mat(LF_MSTmat_MCI);
LF_MCI = [];
LF_MCI = nonzeros(LF_MSTmat_MCI);
% for i=1:size(LF_MSTmat_MCI,1)
%  LF_MCI(i) = mean(nonzeros(LF_MSTmat_MCI(i,:)),1)
% end

%%
for i = 1:size(BC_max_MSTmat_AD, 1)
    for j = 1:size(BC_max_MSTmat_AD, 2)
        if isempty(BC_max_MSTmat_AD{i, j})
            BC_max_MSTmat_AD{i,j} = 0;
        end
    end
end
BC_max_MSTmat_AD = cell2mat(BC_max_MSTmat_AD);
BC_AD = [];
BC_AD = nonzeros(BC_max_MSTmat_AD);
% for i=1:size(BC_max_MSTmat_AD,1)
%  BC_AD(i) = mean(nonzeros(BC_max_MSTmat_AD(i,:)));
% end

for i = 1:size(BC_max_MSTmat_HC, 1)
    for j = 1:size(BC_max_MSTmat_HC, 2)
        if isempty(BC_max_MSTmat_HC{i, j})
            BC_max_MSTmat_HC{i,j} = 0;
        end
    end
end
BC_max_MSTmat_HC = cell2mat(BC_max_MSTmat_HC);
BC_HC = [];
BC_HC = nonzeros(BC_max_MSTmat_HC);
% for i=1:size(BC_max_MSTmat_HC,1)
%  BC_HC(i) = mean(nonzeros(BC_max_MSTmat_HC(i,:)), 1)
% end

for i = 1:size(BC_max_MSTmat_MCI, 1)
    for j = 1:size(BC_max_MSTmat_MCI, 2)
        if isempty(BC_max_MSTmat_MCI{i, j})
            BC_max_MSTmat_MCI{i,j} = 0;
        end
    end
end
BC_max_MSTmat_MCI = cell2mat(BC_max_MSTmat_MCI);
BC_MCI = [];
BC_MCI = nonzeros(BC_max_MSTmat_MCI);
% for i=1:size(BC_max_MSTmat_MCI,1)
%  BC_MCI(i) = mean(nonzeros(BC_max_MSTmat_MCI(i,:)),1)
% end

%%
for i = 1:size(k_MSTmat_AD, 1)
    for j = 1:size(k_MSTmat_AD, 2)
        % if isinf(k_MSTmat_AD{i, j})
        if isempty(k_MSTmat_AD{i, j})
            k_MSTmat_AD{i,j} = 0;
        end
    end
end
k_MSTmat_AD = cell2mat(k_MSTmat_AD);
k_AD = [];
k_AD = nonzeros(k_MSTmat_AD);

for i = 1:size(k_MSTmat_HC, 1)
    for j = 1:size(k_MSTmat_HC, 2)
        % if isinf(k_MSTmat_HC{i, j})
        if isempty(k_MSTmat_HC{i, j})
            k_MSTmat_HC{i,j} = 0;
        end
    end
end
k_MSTmat_HC = cell2mat(k_MSTmat_HC);
k_HC = [];
k_HC = nonzeros(k_MSTmat_HC);

for i = 1:size(k_MSTmat_MCI, 1)
    for j = 1:size(k_MSTmat_MCI, 2)
        % if isinf(k_MSTmat_MCI(i, j))
        if isempty(k_MSTmat_MCI{i, j})
            k_MSTmat_MCI{i,j} = 0;
        end
    end
end
k_MSTmat_MCI = cell2mat(k_MSTmat_MCI);
k_MCI = [];
k_MCI = nonzeros(k_MSTmat_MCI);

%%
for i = 1:size(Ec_MSTmat_AD_m, 1)
    for j = 1:size(Ec_MSTmat_AD_m, 2)
        if isempty(Ec_MSTmat_AD_m{i, j})
            Ec_MSTmat_AD_m{i,j} = 0;
        end
    end
end
Ec_MSTmat_AD_m = cell2mat(Ec_MSTmat_AD_m);
Ec_AD = [];
Ec_AD = nonzeros(Ec_MSTmat_AD_m);

for i = 1:size(Ec_MSTmat_HC_m, 1)
    for j = 1:size(Ec_MSTmat_HC_m, 2)
        if isempty(Ec_MSTmat_HC_m{i, j})
            Ec_MSTmat_HC_m{i,j} = 0;
        end
    end
end
Ec_MSTmat_HC_m = cell2mat(Ec_MSTmat_HC_m);
Ec_HC = [];
Ec_HC = nonzeros(Ec_MSTmat_HC_m);

for i = 1:size(Ec_MSTmat_MCI_m, 1)
    for j = 1:size(Ec_MSTmat_MCI_m, 2)
        if isempty(Ec_MSTmat_MCI_m{i, j})
            Ec_MSTmat_MCI_m{i,j} = 0;
        end
    end
end
Ec_MSTmat_MCI_m = cell2mat(Ec_MSTmat_MCI_m);
Ec_MCI = [];
Ec_MCI = nonzeros(Ec_MSTmat_MCI_m);

%%
for i = 1:size(Deg_MSTmat_AD, 1)
    for j = 1:size(Deg_MSTmat_AD, 2)
        if isempty(Deg_MSTmat_AD{i, j})
            Deg_MSTmat_AD{i,j} = 0;
        end
        Deg_max_AD(i,j) = max(Deg_MSTmat_AD{i,j});
    end
end
Deg_AD = [];
% Deg_AD = nonzeros(Deg_MSTmat_AD);
for i=1:size(Deg_MSTmat_AD,1)
 Deg_AD{i} = nonzeros(Deg_max_AD(i,:));
 Deg_AD_m(i) = mean(Deg_AD{1,i});
end
Deg_AD = Deg_AD_m/size(Deg_AD_m,2);


for i = 1:size(Deg_MSTmat_HC, 1)
    for j = 1:size(Deg_MSTmat_HC, 2)
        if isempty(Deg_MSTmat_HC{i, j})
            Deg_MSTmat_HC{i,j} = 0;
        end
        Deg_max_HC(i,j) = max(Deg_MSTmat_HC{i,j});
    end
end
Deg_HC = [];
for i=1:size(Deg_MSTmat_HC,1)
 Deg_HC{i} = nonzeros(Deg_max_HC(i,:));
 Deg_HC_m(i) = mean(Deg_HC{i});
end
Deg_HC = Deg_HC_m/size(Deg_HC_m,2);


for i = 1:size(Deg_MSTmat_MCI, 1)
    for j = 1:size(Deg_MSTmat_MCI, 2)
        if isempty(Deg_MSTmat_MCI{i, j})
            Deg_MSTmat_MCI{i,j} = 0;
        end
        Deg_max_MCI(i,j) = max(Deg_MSTmat_MCI{i,j});
    end
end
Deg_MCI = [];
for i=1:size(Deg_MSTmat_MCI,1)
 Deg_MCI{i} = nonzeros(Deg_max_MCI(i,:));
 Deg_MCI_m(i) = mean(Deg_MCI{i});
end
Deg_MCI = Deg_MCI_m/size(Deg_MCI_m,2);

%%

[p12, h12, stats12] = ranksum(Th_AD, Th_HC);
[p13, h13, stats13] = ranksum(Th_AD, Th_MCI);
[p23, h23, stats23] = ranksum(Th_HC, Th_MCI);

[p12, h12, stats12] = ranksum(Diam_AD, Diam_HC);
[p13, h13, stats13] = ranksum(Diam_AD, Diam_MCI);
[p23, h23, stats23] = ranksum(Diam_HC, Diam_MCI);

[p12, h12, stats12] = ranksum(LF_AD, LF_HC);
[p13, h13, stats13] = ranksum(LF_AD, LF_MCI);
[p23, h23, stats23] = ranksum(LF_HC,LF_MCI);

[p12, h12, stats12] = ranksum(BC_AD, BC_HC);
[p13, h13, stats13] = ranksum(BC_AD, BC_MCI);
[p23, h23, stats23] = ranksum(BC_HC,BC_MCI);

[p12, h12, stats12] = ranksum(k_AD, k_HC);
[p13, h13, stats13] = ranksum(k_AD, k_MCI);
[p23, h23, stats23] = ranksum(k_HC, k_MCI);

[p12, h12, stats12] = ranksum(Ec_AD, Ec_HC);
[p13, h13, stats13] = ranksum(Ec_AD, Ec_MCI);
[p23, h23, stats23] = ranksum(Ec_HC, Ec_MCI);

[p12, h12, stats12] = ranksum(Deg_AD, Deg_HC);
[p13, h13, stats13] = ranksum(Deg_AD, Deg_MCI);
[p23, h23, stats23] = ranksum(Deg_HC, Deg_MCI);

% Display the p-values and test statistics for all pairwise comparisons
disp(['p-value (AD vs. HC): ', num2str(p12)]);
disp(['Test statistic (AD vs. HC): ', num2str(stats12.ranksum)]);
disp(['p-value (AD vs. MCI): ', num2str(p13)]);
disp(['Test statistic (AD vs. MCI): ', num2str(stats13.ranksum)]);
disp(['p-value (HC vs. MCI): ', num2str(p23)]);
disp(['Test statistic (HC vs. MCI): ', num2str(stats23.ranksum)]);

%% One-way ANOVA
group = [group1, group2, group3];
groupLabels = [ones(1, size(group1,2)), 2*ones(1, size(group2,2)), 3*ones(1, size(group3,2))];  % Group labels: 1, 2, 3
% group = [group1, group2];  %AD vs. HC
% groupLabels = [ones(1, size(group1,2)), 2*ones(1, size(group2,2))];
% group = [group1, group3];  %AD vs. MCI
% groupLabels = [ones(1, size(group1,2)), 2*ones(1, size(group3,2))];
[p, tbl, stats] = anova1(group, groupLabels');
% Display the results
disp(['p-value: ', num2str(p)]);
disp(tbl);
disp(stats);

%% Distances between static subgraphs of 3 groups
% [Mreordered_ADMCI, Mindices_ADMCI, cost_ADMCI] = align_matrices(meanMST_AD_all,meanMST_MCI_all,'cosang',1);
% [Mreordered_ADHC, Mindices_ADHC, cost_ADHC] = align_matrices(meanMST_AD_all,meanMST_HC_all,'cosang',1);
% [Mreordered_HCMCI, Mindices_HCMCI, cost_HCMCI] = align_matrices(meanMST_HC_all,meanMST_MCI_all,'cosang',1);
% [Mreordered_ADMCI, Mindices_ADMCI, cost_ADMCI] = align_matrices(meanMST_AD_all,meanMST_MCI_all,'sqrdff',1);
% [Mreordered_ADHC, Mindices_ADHC, cost_ADHC] = align_matrices(meanMST_AD_all,meanMST_HC_all,'sqrdff',1);
% [Mreordered_HCMCI, Mindices_HCMCI, cost_HCMCI] = align_matrices(meanMST_HC_all,meanMST_MCI_all,'sqrdff',1);
% [Mreordered_ADMCI, Mindices_ADMCI, cost_ADMCI] = align_matrices(meanMST_AD_all,meanMST_MCI_all,'absdff',1);
% [Mreordered_ADHC, Mindices_ADHC, cost_ADHC] = align_matrices(meanMST_AD_all,meanMST_HC_all,'absdff',1);
% [Mreordered_HCMCI, Mindices_HCMCI, cost_HCMCI] = align_matrices(meanMST_HC_all,meanMST_MCI_all,'absdff',1);
% cost_ADMCI
% cost_ADHC
% cost_HCMCI
% figure, imagesc(Mreordered_HCMCI)
% Sim_Jac_ADMCI = sum(meanMST_AD_all & meanMST_MCI_all, 'all') / sum(meanMST_AD_all |meanMST_MCI_all, 'all'); %Jaccard similarity
% Sim_Jac_ADHC = sum(meanMST_AD_all & meanMST_HC_all, 'all') / sum(meanMST_AD_all |meanMST_HC_all, 'all'); %Jaccard similarity
for i=1:size(meanMST_AD_all,1)
    for j=1:size(meanMST_MCI_all,1)
        % Sim_Jac_ADMCI(i,j) = sum(meanMST_AD_all(i,:) & meanMST_MCI_all(j,:), 'all') / sum(meanMST_AD_all(i,:) |meanMST_MCI_all(j,:), 'all'); %Jaccard similarity
        % Sim_Jac_ADHC(i,j) = sum(meanMST_AD_all(i,:) & meanMST_HC_all(j,:), 'all') / sum(meanMST_AD_all(i,:) |meanMST_HC_all(j,:), 'all'); %Jaccard similarity
        % Sim_Jac_MCIHC(i,j) = sum(meanMST_MCI_all(i,:) & meanMST_HC_all(j,:), 'all') / sum(meanMST_MCI_all(i,:) |meanMST_HC_all(j,:), 'all'); %Jaccard similarity
        % Sim_cos_ADMCI(i,j) = dot(meanMST_AD_all(i,:), meanMST_MCI_all(j,:), 2)/ (norm(meanMST_AD_all(i,:), 2) * norm(meanMST_MCI_all(j,:), 2)); %Cosine similarity
        % Sim_cos_ADHC(i,j) = dot(meanMST_AD_all(i,:), meanMST_HC_all(j,:), 2)/ (norm(meanMST_AD_all(i,:), 2) * norm(meanMST_HC_all(j,:), 2)); %Cosine similarity
        % Sim_cos_MCIHC(i,j) = dot(meanMST_MCI_all(i,:), meanMST_HC_all(j,:), 2)/ (norm(meanMST_MCI_all(i,:), 2) * norm(meanMST_HC_all(j,:), 2)); %Cosine similarity
        % diss_MCIHC(i,j) = inf_dissimilarity(meanMST_MCI_all(i,:), meanMST_HC_all(j,:));
    end
end
diss_MCIHC = inf_dissimilarity(meanMST_MCI_all, meanMST_HC_all)
diss_ADHC = inf_dissimilarity(meanMST_AD_all, meanMST_HC_all)
diss_ADMCI = inf_dissimilarity(meanMST_AD_all, meanMST_MCI_all)


% Wilcoxon Test Between Static Subgraphs
% [p_ADMCI, h_ADMCI] = signrank(meanMST_AD_all(:), meanMST_MCI_all(:));
% [p_ADHC, h_ADHC] = signrank(meanMST_AD_all(:), meanMST_HC_all(:));
% [p_MCIHC, h_MCIHC] = signrank(meanMST_MCI_all(:), meanMST_HC_all(:));
[p_ADMCI, h_ADMCI] = ranksum(meanMST_AD_all(:), meanMST_MCI_all(:));
[p_ADHC, h_ADHC] = ranksum(meanMST_AD_all(:), meanMST_HC_all(:));
[p_MCIHC, h_MCIHC] = ranksum(meanMST_MCI_all(:), meanMST_HC_all(:));
disp(['p-value (two-tailed) of ADMCI: ', num2str(p_ADMCI)])
% disp('h = 1 (reject null hypothesis) or h = 0 (fail to reject)')
disp(h_ADMCI)
disp(['p-value (two-tailed) of ADHC: ', num2str(p_ADHC)])
disp(['p-value (two-tailed) of HCMCI: ', num2str(p_MCIHC)])
disp(h_ADHC)
disp(h_MCIHC)

%% Classifcation
for i = 1:size(Ec_MSTmat_MCI, 1)
    for j = 1:size(Ec_MSTmat_MCI, 2)
        if isempty(Ec_MSTmat_MCI{i, j})
            Ec_MSTmat_MCI{i, j} = zeros(1,148);
        end
    end
end
for i = 1:size(Deg_MSTmat_MCI, 1)
    for j = 1:size(Deg_MSTmat_MCI, 2)
        if isempty(Deg_MSTmat_MCI{i, j})
            Deg_MSTmat_MCI{i, j} = zeros(1,148);
        end
    end
end
for i = 1:size(Th_MSTmat_MCI, 1)
    for j = 1:size(Th_MSTmat_MCI, 2)
        if isempty(Th_MSTmat_MCI{i, j})
            Th_MSTmat_MCI{i, j} = 0;
        end
    end
end
for i = 1:size(Diam_MSTmat_MCI, 1)
    for j = 1:size(Diam_MSTmat_MCI, 2)
        if isempty(Diam_MSTmat_MCI{i, j})
            Diam_MSTmat_MCI{i, j} = 0;
        end
    end
end
for i = 1:size(LF_MSTmat_MCI, 1)
    for j = 1:size(LF_MSTmat_MCI, 2)
        if isempty(LF_MSTmat_MCI{i, j})
            LF_MSTmat_MCI{i, j} = 0;
        end
    end
end
for i = 1:size(BC_max_MSTmat_MCI, 1)
    for j = 1:size(BC_max_MSTmat_MCI, 2)
        if isempty(BC_max_MSTmat_MCI{i, j})
            BC_max_MSTmat_MCI{i, j} = 0;
        end
    end
end
for i = 1:size(k_MSTmat_MCI, 1)
    for j = 1:size(k_MSTmat_MCI, 2)
        if isempty(k_MSTmat_MCI{i, j})
            k_MSTmat_MCI{i, j} = 0;
        end
    end
end
% Feat_AD = [Th_MSTmat_AD   Diam_MSTmat_AD  LF_MSTmat_AD   BC_max_MSTmat_AD   k_MSTmat_AD   cell2mat(Deg_MSTmat_AD)  cell2mat(Ec_MSTmat_AD)];
Feat_AD = [cell2mat(Th_MSTmat_AD)   cell2mat(Diam_MSTmat_AD)  cell2mat(LF_MSTmat_AD)   cell2mat(BC_max_MSTmat_AD)   cell2mat(k_MSTmat_AD)   cell2mat(Deg_MSTmat_AD)  cell2mat(Ec_MSTmat_AD)];
Feat_HC = [cell2mat(Th_MSTmat_HC)   cell2mat(Diam_MSTmat_HC)  cell2mat(LF_MSTmat_HC)   cell2mat(BC_max_MSTmat_HC)   cell2mat(k_MSTmat_HC)   cell2mat(Deg_MSTmat_HC)  cell2mat(Ec_MSTmat_HC)];
Feat_MCI = [cell2mat(Th_MSTmat_MCI)   cell2mat(Diam_MSTmat_MCI)  cell2mat(LF_MSTmat_MCI)   cell2mat(BC_max_MSTmat_MCI)   cell2mat(k_MSTmat_MCI)   cell2mat(Deg_MSTmat_MCI)  cell2mat(Ec_MSTmat_MCI)];

% Feat_AD = [cell2mat(Deg_MSTmat_AD)  cell2mat(Ec_MSTmat_AD)];
% Feat_HC = [cell2mat(Deg_MSTmat_HC)  cell2mat(Ec_MSTmat_HC)];
% Feat_MCI = [cell2mat(Deg_MSTmat_MCI)  cell2mat(Ec_MSTmat_MCI)];

Feat_MCI = [Feat_MCI  zeros(size(Feat_MCI, 1), size(Feat_HC,2)-size(Feat_MCI, 2))];
group = [Feat_AD; Feat_HC; Feat_MCI];
groupLabels = [ones(1, size(Feat_AD,1)), 2*ones(1, size(Feat_HC,1)), 3*ones(1, size(Feat_MCI,1))]';
if isnumeric(groupLabels)
    labels = categorical(groupLabels);
end
[idx,scores] = fscmrmr(group,groupLabels);
% num_sel = 1000;
% features = group(:, idx(1:num_sel));
features = group(:, idx);

cvp = cvpartition(groupLabels, 'KFold',10)

% SVM_Mdl = fitcecoc(Xtrain, ytrain);
SVM_Mdl = fitcecoc(features, groupLabels, 'CVPartition',cvp);
SVMRate = kfoldLoss(SVM_Mdl)
% Ypred_SVM = predict(SVM_Mdl, Xtest);
% Conf_svm = confusionmat(ytest, Ypred_SVM);
% error = resubLoss(SVM_Mdl)
% L = loss(SVM_Mdl,Xtest,ytest)

k = 10;
% KNN_Mdl = fitcknn(Xtrain, ytrain,'NumNeighbors',k,...
%     'NSMethod','exhaustive',...
%     'Distance','cityblock',...
%     'DistanceWeight','squaredinverse',...
%     'Standardize',true);
% Ypred_KNN = KNN_Mdl.predict(Xtest);
% loss_KNN = resubLoss(KNN_Mdl)
KNNCVModel = fitcknn(features, groupLabels,'CVPartition',cvp);
KNNRate = kfoldLoss(KNNCVModel)

% LDA_Mdl = fitcdiscr(Xtrain, ytrain);
% Ypred_LDA = LDA_Mdl.predict(Xtest);
% loss_LDA = resubLoss(LDA_Mdl)
discrCVModel = fitcdiscr(features, groupLabels,'CVPartition',cvp);
discrRate = kfoldLoss(discrCVModel)

% DTREE_Mdl = fitctree(Xtrain, ytrain);
% Ypred_DTREE = DTREE_Mdl.predict(Xtest);
% loss_DTREE = resubLoss(DTREE_Mdl)
treeCVModel = fitctree(features, groupLabels,'CVPartition',cvp);
treeRate = kfoldLoss(treeCVModel)

% BT_Mdl = TreeBagger(50,Xtrain, ytrain,...
%     Method="classification",...
%     OOBPrediction="on")
% Ypred_BT = BT_Mdl.predict(Xtest);
% % loss_BT = resubLoss(BT_Mdl)
%

%% Define Neural Network Classifier Architecture
% num_features = size(features, 2);  % Number of input features
% num_classes = numel(unique(labels)); % Number of output classes (unique categories)

% Specify hidden layer structure
% hiddenLayerSizes = [256 128];  % Adjust hidden layer sizes as needed

% Define training options (corrected for compatibility with older versions)
% Define network architecture with hidden layers
% net = feedforwardnet;
% Split data into training and test sets (e.g., 70% for training, 30% for testing)
cv = cvpartition(size(features, 1), 'Holdout', 0.3);
trainIdx = training(cv);
testIdx = test(cv);

trainFeatures = features(trainIdx, :);
testFeatures = features(testIdx, :);
trainLabels = groupLabels(trainIdx, :);
testLabels = groupLabels(testIdx, :);
net = feedforwardnet([2]);
net.inputs{1}.processFcns
net.outputs{2}.processFcns

net = train(net, trainFeatures', trainLabels');

% Predict outputs on test data
testOutputs = sim(net, testFeatures');
% Convert predicted outputs to class labels
predictedLabels = round(testOutputs);  % Assumes one-hot encoded labels
% Calculate accuracy
accuracy = sum(predictedLabels == testLabels) / size(testLabels, 1);
disp(['Accuracy on test data: ', num2str(accuracy)]);
% Train the network with the defined layers
% net = trainNetwork(features, labels, layers, trainOptions);


%%
for i = 1:size(Diam_MSTmat_AD, 1)
    for j = 1:size(Diam_MSTmat_AD, 2)
        if Diam_MSTmat_AD{i, j}==0;
            Diam_MSTmat_AD{i, j} = [];
        end
    end
end

for i = 1:size(Diam_MSTmat_AD, 1)
    for j = 1:size(Diam_MSTmat_AD, 2)
        if ~isempty(Diam_MSTmat_AD{i, j})
            Diam_MSTmat_AD_m(i) = mean(cell2mat(Diam_MSTmat_AD(i, :)));
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:size(Th_MSTmat_AD, 1)
    for j = 1:size(Th_MSTmat_AD, 2)
        if Th_MSTmat_AD{i, j}==0;
            Th_MSTmat_AD{i, j} = [];
        end
    end
end

for i = 1:size(Th_MSTmat_AD, 1)
    for j = 1:size(Th_MSTmat_AD, 2)
        if ~isempty(Th_MSTmat_AD{i, j})
            Th_MSTmat_AD_m(i) = mean(cell2mat(Th_MSTmat_AD(i, :)));
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:size(k_MSTmat_AD, 1)
    for j = 1:size(k_MSTmat_AD, 2)
        if k_MSTmat_AD{i, j}==0;
            k_MSTmat_AD{i, j} = [];
        end
    end
end

for i = 1:size(k_MSTmat_AD, 1)
    for j = 1:size(k_MSTmat_AD, 2)
        if ~isempty(k_MSTmat_AD{i, j})
            k_MSTmat_AD_m(i) = mean(cell2mat(k_MSTmat_AD(i, :)));
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:size(BC_max_MSTmat_AD, 1)
    for j = 1:size(BC_max_MSTmat_AD, 2)
        if BC_max_MSTmat_AD{i, j}==0;
            BC_max_MSTmat_AD{i, j} = [];
        end
    end
end

for i = 1:size(BC_max_MSTmat_AD, 1)
    for j = 1:size(BC_max_MSTmat_AD, 2)
        if ~isempty(BC_max_MSTmat_AD{i, j})
            BC_max_MSTmat_AD_m(i) = mean(cell2mat(BC_max_MSTmat_AD(i, :)));
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:size(Ec_MSTmat_AD, 1)
    for j = 1:size(Ec_MSTmat_AD, 2)
        if Ec_MSTmat_AD{i, j}==0;
            Ec_MSTmat_AD{i, j} = [];
        end
    end
end

for i = 1:size(Ec_MSTmat_AD, 1)
    for j = 1:size(Ec_MSTmat_AD, 2)
        if ~isempty(Ec_MSTmat_AD{i, j})
            Ec_MSTmat_AD_m{i,j} = mean((Ec_MSTmat_AD{i, j}));
        end
    end
end

for i = 1:size(Ec_MSTmat_AD_m, 1)
    for j = 1:size(Ec_MSTmat_AD_m, 2)
        if Ec_MSTmat_AD_m{i, j}==0
            Ec_MSTmat_AD_m{i, j} = [];
        end
    end
end

for i = 1:size(Ec_MSTmat_AD_m, 1)
  Ec_MSTmat_AD_m_m(i) = mean(cell2mat(Ec_MSTmat_AD_m(i, :)));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:size(Deg_MSTmat_AD, 1)
    for j = 1:size(Deg_MSTmat_AD, 2)
        if Deg_MSTmat_AD{i, j}==0;
            Deg_MSTmat_AD{i, j} = [];
        end
    end
end

for i = 1:size(Deg_MSTmat_AD, 1)
    for j = 1:size(Deg_MSTmat_AD, 2)
        if ~isempty(Deg_MSTmat_AD{i, j})
            Deg_MSTmat_AD_m{i,j} = mean((Deg_MSTmat_AD{i, j}));
        end
    end
end

for i = 1:size(Deg_MSTmat_AD_m, 1)
    for j = 1:size(Deg_MSTmat_AD_m, 2)
        if Deg_MSTmat_AD_m{i, j}==0
            Deg_MSTmat_AD_m{i, j} = [];
        end
    end
end

for i = 1:size(Deg_MSTmat_AD_m, 1)
  Deg_MSTmat_AD_m_m(i) = mean(cell2mat(Deg_MSTmat_AD_m(i, :)));
end

