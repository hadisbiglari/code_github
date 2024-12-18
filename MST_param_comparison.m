%% comparison
%% Tree Hierarchy
for i =1:size(Treehierarchy_MCI, 2)
    mean_Treehierarchy_MCI(i)=mean(Treehierarchy_MCI{i}(1));
    % std_Treehierarchy_MCI(i)=std(Treehierarchy_MCI{i});
end
TH_MCI_m = mean(mean_Treehierarchy_MCI)
TH_MCI_s = std(mean_Treehierarchy_MCI)


for i =1:size(Treehierarchy, 2)
    mean_Treehierarchy_HC(i)=mean(Treehierarchy{i});
end
TH_HC_m = mean(mean_Treehierarchy_HC)
TH_HC_s = std(mean_Treehierarchy_HC)

for i =1:size(Treehierarchy_AD, 2)
    mean_Treehierarchy_AD(i)=mean(Treehierarchy_AD{i});
end
TH_AD_m = mean(mean_Treehierarchy_AD)
TH_AD_s = std(mean_Treehierarchy_AD)

%% Diameter
diameter_MCI_m = mean(diameter_MCI)
diameter_AD_m = mean(diameter_AD)
diameter_HC_m = mean(diameter)
diameter_MCI_s = std(diameter_MCI)
diameter_AD_s = std(diameter_AD)
diameter_HC_s = std(diameter)

%% Leaf Fraction
LF_MCI_m = mean(cell2mat(leaf_fraction_MSTMCI))
LF_HC_m = mean(cell2mat(leaf_fraction_MSTHC))
LF_AD_m = mean(cell2mat(leaf_fraction_MSTAD))
LF_MCI_s = std(cell2mat(leaf_fraction_MSTMCI))
LF_HC_s = std(cell2mat(leaf_fraction_MSTHC))
LF_AD_s = std(cell2mat(leaf_fraction_MSTAD))

%% Betweenness Centrality
N=148;
for i =1:size(Treehierarchy_AD,2)
    max_BC_AD(i)=max(BC_MSTAD{i});
end
BC_max_AD = max(max_BC_AD)/((N-1)*(N-2))
BC_max_AD_s = std(max_BC_AD)/((N-1)*(N-2))

for i =1:size(Treehierarchy,2)
    max_BC_HC(i)=max(BC_MSTHC{i});
end
BC_max_HC = max(max_BC_HC)/((N-1)*(N-2))
BC_max_HC_s = std(max_BC_HC)/((N-1)*(N-2))

for i =1:size(Treehierarchy_MCI,2)
    max_BC_MCI(i)=max(BC_MSTMCI{i});
end
BC_max_MCI = max(max_BC_MCI)/((N-1)*(N-2))
BC_max_MCI_s = std(max_BC_MCI)/((N-1)*(N-2))

%% Maximum Degree
for i =1:size(Treehierarchy_AD,2)
    max_eccentricity_AD(i)=max(deg_AD{i});
end
Deg_max_AD = max(max_eccentricity_AD)/size(Treehierarchy_AD,2)
Deg_max_AD_s = std(max_eccentricity_AD)/size(Treehierarchy_AD,2)

for i =1:size(Treehierarchy,2)
    max_deg_HC(i)=max(deg_HC{i});
end
Deg_max_HC = max(max_deg_HC)/size(Treehierarchy,2)
Deg_max_HC_s = std(max_deg_HC)/size(Treehierarchy,2)

for i =1:size(Treehierarchy_MCI,2)
    max_deg_MCI(i)=max(deg_MCI{i});
end
Deg_max_MCI = max(max_deg_MCI)/size(Treehierarchy_MCI,2)
Deg_max_MCI_s = std(max_deg_MCI)/size(Treehierarchy_MCI,2)

%% Kappa
kappa_AD_mean
kappa_HC_mean
kappa_MCI_mean
kappa_AD_m
kappa_HC_m
kappa_MCI_m
kappa_AD_s = std(kappa_AD)
kappa_HC_s = std(kappa_HC)
kappa_MCI_s = std(kappa_MCI)

%% Eccentricity
for i =1:size(Treehierarchy_AD,2)
    max_eccentricity_AD(i)=max(eccentricity_AD{i});
end
eccentricity_AD_tot = max(max_eccentricity_AD)/size(eccentricity_AD,2)
eccentricity_AD_tot_s = std(max_eccentricity_AD)/size(eccentricity_AD,2)

for i =1:size(Treehierarchy,2)
    max_eccentricity_HC(i)=max(eccentricity{i});
end
eccentricity_HC_tot = max(max_eccentricity_HC)/size(eccentricity,2)
eccentricity_HC_tot_s = std(max_eccentricity_HC)/size(eccentricity,2)

for i =1:size(eccentricity_MCI,2)
    max_eccentricity_MCI(i)=max(eccentricity_MCI{i});
end
eccentricity_MCI_tot = max(max_eccentricity_MCI)/size(eccentricity_MCI,2)
eccentricity_MCI_tot_s = std(max_eccentricity_MCI)/size(eccentricity_MCI,2)

%% Wilcoxon signed-rank test
% group1=[TH_AD_m  diameter_AD_m  LF_AD_m  BC_max_AD  Deg_max_AD  kappa_AD_m  eccentricity_AD_tot];
% group2=[TH_HC_m  diameter_HC_m  LF_HC_m  BC_max_HC  Deg_max_HC  kappa_HC_m  eccentricity_HC_tot];
% group3=[TH_MCI_m  diameter_MCI_m  LF_MCI_m  BC_max_MCI  Deg_max_MCI  kappa_MCI_m   eccentricity_MCI_tot];

group1=[Treehierarchy_AD_tot   diameterAD_m_tot  LF_AD_tot   BC_AD_tot'  deg_AD_tot  kappa_AD_tot  eccentricity_AD_tot];
group2=[Treehierarchy_HC_tot   diameterHC_m_tot  LF_HC_tot   BC_HC_tot'  deg_HC_tot  kappa_HC_tot  eccentricity_HC_tot];
group3=[Treehierarchy_MCI_tot   diameterMCI_m_tot  LF_MCI_tot   BC_MCI_tot'  deg_MCI_tot  kappa_MCI_tot   eccentricity_MCI_tot];

% [p12, h12, stats12] = ranksum(group1, group2);
[p13, h13, stats13] = ranksum(group1, group3);
[p23, h23, stats23] = ranksum(group2, group3);

[p12, h12, stats12] = signrank(group1, group2);
% [p13, h13, stats13] = signrank(group1, group3);
% [p23, h23, stats23] = signrank(group2, group3);

% Display the p-values and test statistics for all pairwise comparisons
disp(['p-value (AD vs. HC): ', num2str(p12)]);
% disp(['Test statistic (AD vs. HC): ', num2str(stats12.ranksum)]);
disp(['Test statistic (AD vs. HC): ', num2str(stats12.signedrank)]);
disp(['p-value (AD vs. MCI): ', num2str(p13)]);
disp(['Test statistic (AD vs. MCI): ', num2str(stats13.ranksum)]);
% disp(['Test statistic (AD vs. MCI): ', num2str(stats13.signedrank)]);
disp(['p-value (HC vs. MCI): ', num2str(p23)]);
disp(['Test statistic (HC vs. MCI): ', num2str(stats23.ranksum)]);
% disp(['Test statistic (HC vs. MCI): ', num2str(stats23.signedrank)]);

%% One-way ANOVA
% group3 = [group3 0 0 0];
group = [group1, group2, group3];
groupLabels = [ones(1, size(group1,2)), 2*ones(1, size(group2,2)), 3*ones(1, size(group3,2))];  % Group labels: 1, 2, 3
[p, tbl, stats] = anova1(group, groupLabels);
% Display the results
disp(['p-value: ', num2str(p)]);
disp(tbl);
disp(stats);

% %% Classifcation
% group = [group1; group2; group3];
% group = zscore(group);
% cvp = cvpartition(groupLabels, 'holdout', floor(size(group, 2) * 0.25));
% Xtrain = group(cvp.training);
% ytrain = groupLabels(cvp.training);
% Xtest  = group(cvp.test);
% ytest  = groupLabels(cvp.test);
% 
% 
% SVM_Mdl = fitcecoc(Xtrain, ytrain);
% Ypred_SVM = predict(SVM_Mdl, Xtest);
% % Conf_svm = confusionmat(ytest, Ypred_SVM);
% error = resubLoss(SVM_Mdl)
% L = loss(SVM_Mdl,Xtest,ytest)
% 
% k = 10;
% KNN_Mdl = fitcknn(Xtrain, ytrain,'NumNeighbors',k,...
%     'NSMethod','exhaustive',...
%     'Distance','cityblock',...
%     'DistanceWeight','squaredinverse',...
%     'Standardize',true);
% Ypred_KNN = KNN_Mdl.predict(Xtest);
% loss_KNN = resubLoss(KNN_Mdl)
% 
% 
% LDA_Mdl = fitcdiscr(Xtrain, ytrain);
% Ypred_LDA = LDA_Mdl.predict(Xtest);
% loss_LDA = resubLoss(LDA_Mdl)
% 
% 
% DTREE_Mdl = fitctree(Xtrain, ytrain);
% Ypred_DTREE = DTREE_Mdl.predict(Xtest);
% loss_DTREE = resubLoss(DTREE_Mdl)
% 
% 
% BT_Mdl = TreeBagger(50,Xtrain, ytrain,...
%     Method="classification",...
%     OOBPrediction="on")
% Ypred_BT = BT_Mdl.predict(Xtest);
% % loss_BT = loss(BT_Mdl, ytest', Ypred_BT)

