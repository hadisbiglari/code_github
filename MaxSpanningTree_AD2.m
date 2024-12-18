%% Maximum Spanning Tree calculation for AD whole recordings
% clear;
clc
% close all;
addpath('C:/Users/MIQDAD/Desktop/PhD/PROPOSED Method/Graph analysis_MST/Code');
addpath('C:/Users/MIQDAD/Desktop/PhD/PROPOSED Method/Graph analysis_MST/Code/2019_03_03_BCT')
addpath('C:/Users/MIQDAD/Desktop/PhD/PROPOSED Method/Graph analysis_MST/Code/FCLAB-master/FCLAB-master/FCLAB1.0.0/FC_metrics')
addpath('C:/Users/MIQDAD/Desktop/PhD/PROPOSED Method/Graph analysis_MST/Code/FCLAB-master/FCLAB-master/FCLAB1.0.0/MST_params')
load('EEGAD.mat')        %% EEGAD = EEG(1:36); EEGHC = EEG(37:65); EEGFD = EEG(66:88);
% load('PLIAD_eeg.mat')
% ConnAD_all = PLIAD_eeg;
% load('MEGAD.mat')
% load('PLIAD.mat')
% load('plvAD_all.mat')
% load('WPLIAD2.mat')
load('WPLIAD_eeg2.mat')
% ConnAD_all = WPLIAD2;
ConnAD_all = WPLIAD_eeg2;
% fs = 678.19;
% longest data = 29x1695xfs = 72.48 sec
% shortest data = 4x1695xfs = 10 sec

%% Find Static graph for AD
%% Over each subjest across time
%%%%%%%%%%% Graph of all frames of all subjects %%%%%%%%%%%%%%%
for k=1:size(ConnAD_all,1)
    for t=1:size(ConnAD_all,2)
        if ~isempty(ConnAD_all{k,t})
            % figure, imagesc(ConnAD_all{k,t}), title('Connectivity matrix')
            S_AD = [];
            t_AD=[];
            for i = 1:size(ConnAD_all{1},1)
                start_node = i*ones(size(ConnAD_all{1},1)-i, 1);
                end_node = i+1:size(ConnAD_all{1},1);
                % Append the start and end nodes to the S and t vectors
                S_AD = [S_AD start_node'];
                t_AD = [t_AD end_node];
            end
            for j=1:size(S_AD,2)
                weights_AD_inv_all{k,t}(j)=1./(ConnAD_all{k,t}(S_AD(j),t_AD(j)));
            end
            % Convert the matrix to a graph object
            G_AD = graph(S_AD,t_AD,weights_AD_inv_all{k,t});
            G_AD.Edges;
            k;
            tree_AD{k,t} = minspantree(G_AD);   % MST of AD graphs
            tree_AD{k,t}.Edges;
            weights_AD = tree_AD{k}.Edges.Weight;
            weights_AD = (weights_AD).^-1;
            % figure, plot(G_AD), title('MST graph of AD')
            % highlight(p,tree_AD{k})
            MSTmat_AD{k,t} = adjacency(tree_AD{k,t}, double(weights_AD));    %Convert MST graph to MST matrix
            % figure, imagesc(MSTmat_AD{k}), title('MST matrix of AD')
        end
    end
end

%% Senarios for static graph calculation:
%% Senario 1: mean of MSTmat_AD over each subject (t) and then over all of subjects (k)
MSTmat_AD_all = cell(size(MSTmat_AD,1),1);
flag_AD=zeros(size(MSTmat_AD,1),1);
for k=1:size(MSTmat_AD,1)
    MSTmat_AD_all{k} = zeros(size(MSTmat_AD{1}));
    for t=1:size(MSTmat_AD,2)
        if ~isempty(MSTmat_AD{k,t})
            flag_AD(k)=flag_AD(k)+1;
            currentMatrix_AD_all = MSTmat_AD{k,t};
            MSTmat_AD_all{k} = MSTmat_AD_all{k} + currentMatrix_AD_all;
        end
    end
    meanMSTsubj_AD{k,1} = MSTmat_AD_all{k,1} / (flag_AD(k));
    % figure,imagesc(OccurMatrix_AD_all{k}),title('meanMatrix AD for each
    % subject'); % subject 14 seems noisy
end
for k=1:size(meanMSTsubj_AD,1)
    curr = meanMSTsubj_AD{k};
    sumMST_AD_all = meanMSTsubj_AD{k} + curr;
end
meanMST_AD_all = sumMST_AD_all/size(meanMSTsubj_AD,1);
% Normalize elements using min-max scaling
min_val = min(meanMST_AD_all(:));
max_val = max(meanMST_AD_all(:));
meanMST_AD_all = (meanMST_AD_all - min_val) ./ (max_val - min_val);

thr = max(max(meanMST_AD_all))*0.25;
meanMST_AD_all_thr = meanMST_AD_all.*(meanMST_AD_all>thr);
figure, imagesc(meanMST_AD_all), title('Disease specific matrix of AD'),xlabel('Channels'),ylabel('Channels')
figure, imagesc(meanMST_AD_all_thr), title('Disease specific matrix of AD after thresholding'), xlabel('Channels'), ylabel('Channels')
dlmwrite('Staticgrah_AD.txt', meanMST_AD_all, 'delimiter','\t')
dlmwrite('Staticgrah_AD_thr.txt', meanMST_AD_all_thr, 'delimiter','\t')

% Static_diff_HCAD = meanMST_HC_all_thr - meanMST_AD_all_thr;
% figure, imagesc(Static_diff_HCAD), title('Static matrix difference between HC and AD'),xlabel('Channels'),ylabel('Channels')
% dlmwrite('Static_diff_HCAD.txt', Static_diff_HCAD, 'delimiter','\t')
% Static_diff_HCMCI = meanMST_HC_all_thr - meanMST_MCI_all_thr;
% figure, imagesc(Static_diff_HCMCI), title('Static matrix difference between HC and MCI'),xlabel('Channels'),ylabel('Channels')
% dlmwrite('Static_diff_HCMCI.txt', Static_diff_HCMCI, 'delimiter','\t')
% Static_diff_MCIAD = meanMST_MCI_all_thr - meanMST_AD_all_thr;
% figure, imagesc(Static_diff_MCIAD), title('Static matrix difference between MCI and AD'),xlabel('Channels'),ylabel('Channels')
% dlmwrite('Static_diff_MCIAD.txt', Static_diff_MCIAD, 'delimiter','\t')

%% Senario 2: mean of connectivity matrices over t and then k, then calculate MST for it
%%%%%%%%%%% MST of mean of Occurence matrices (This one is a better refernce for static subgraph calculation) %%%%%%%%%%%%%%
%%%% Ocuurence matrix is defined as summation of all Connectivities for each subject
OccurMatrix_AD_all = cell(size(ConnAD_all,1),1);
for k=1:size(ConnAD_all,1)
    OccurMatrix_AD_all{k} = zeros(size(ConnAD_all{1}));
    for t=1:size(ConnAD_all,2)
        if ~isempty(ConnAD_all{k,t})
            currentMatrix_AD_all = ConnAD_all{k,t};
            OccurMatrix_AD_all{k} = OccurMatrix_AD_all{k} + currentMatrix_AD_all;
        end
    end
    meanConn_AD{k,1} = OccurMatrix_AD_all{k,1} / (flag_AD(k));  % Assuming you have 36 matrices in the cell array
    % figure,imagesc(meanMatrix_HC_all{k}),title('meanMatrix HC for each subject');
end

for k=1:size(meanConn_AD,1)
    curre = meanConn_AD{k};
    sumConn_AD_all = meanConn_AD{k} + curre;
end
meanConn_AD_all = sumConn_AD_all/size(meanConn_AD,1);
% MST for mean connectivity over time (t)
for k=1:size(meanConn_AD,1)
    for j = 1:size(S_AD,2)
        weightsmean_AD_inv{k}(j)=1./(meanConn_AD{k}(S_AD(j),t_AD(j)));
    end
    Gmean_AD = graph(S_AD,t_AD,weightsmean_AD_inv{k});
    Gmean_AD.Edges;
    treemean_AD{k} = minspantree(Gmean_AD);
    treemean_AD{k}.Edges;
    for i = 1:size(meanConn_AD_all, 1)-1
        treemean_AD_nod = treemean_AD{k}.Edges.EndNodes;
        a = treemean_AD_nod(:,1);
        b = treemean_AD_nod(:,2);
        treemean_AD_wht = treemean_AD{k}.Edges.Weight;
        MSTAD_mean_all{k}(a(i),b(i)) = 1/treemean_AD_wht(i);
    end
    MSTAD_mean_all{k} = padarray(MSTAD_mean_all{k},[size(ConnAD_all{1}, 1)-size(MSTAD_mean_all{k},1) 0],0,'post');
    MSTAD_meann_all{k} = triu(MSTAD_mean_all{k})+tril(transpose(MSTAD_mean_all{k}));
end
% MST for mean connectivity over time and then over subjects
for j = 1:size(S_AD,2)
    weightsmean_AD_inv_all(j)=1./(meanConn_AD_all(S_AD(j),t_AD(j)));
end
Gmean_AD_all = graph(S_AD,t_AD,weightsmean_AD_inv_all);
Gmean_AD_all.Edges;
treemean_AD_all = minspantree(Gmean_AD_all);
treemean_AD_all.Edges;
weights_AD_all = treemean_AD_all.Edges.Weight;
weights_AD_all = (weights_AD_all).^-1;
% figure, plot(Gmean_HC_all), title('MST graph of mean of all connectivities')
% highlight(p,treemean_HC_all)
MSTmat_AD_all = adjacency(treemean_AD_all, double(weights_AD_all));    %Convert MST graph to MST matrix
thr2 = max(max(full(MSTmat_AD_all)))*0.25;
MSTmat_AD_all_thr = MSTmat_AD_all.*(MSTmat_AD_all>thr2);
figure, imagesc(MSTmat_AD_all), title('MST matrix of mean of all connectivities for AD')
figure, imagesc(MSTmat_AD_all_thr), title('MST matrix of mean of all connectivities for AD after threshold')

%% Overlap Percentage between mean MST matrix and each MST over time for each Subject %%%%%%%%%%%%%
numSubjects_AD = size(MSTmat_AD, 1);
% MSTsubjects_AD = reshape(cell2mat(MSTAD_all(:)).',148,148,numSubjects_AD);
MSTref_AD = logical(MSTmat_AD_all);
% MSTref_AD = logical(meanMST_AD_all);
for k=1:size(MSTAD_meann_all,2)
    % MSTref_AD{k} = MSTAD_meann_all{k}; % one ref for each subject
    overlapPercentages_AD{k} = zeros(1, numSubjects_AD);
    for l = 1:size(MSTmat_AD, 2)
        if ~isempty(cell2mat(MSTmat_AD(k,l)))
            MSTsubjects_AD{k,l} = logical(MSTmat_AD{k,l});
            % overlapMatrix_AD{k} = MSTref_AD{k} & MSTsubjects_AD(:, :, i);
            overlapMatrix_AD{k} = MSTref_AD & MSTsubjects_AD{k,l};        % one ref for each subject
            % overlapMatrix_AD{k,l} = MSTref_AD & cell2mat(MSTsubjects_AD(k,l));      %for MSTref_AD = Occurmat_tot
            overlapPercentage_AD{k} = sum(full(overlapMatrix_AD{k}(:))) / sum(full(MSTref_AD(:))) * 100;         % one ref for each subject
            % overlapPercentage_AD{k,l} = sum(overlapMatrix_AD{k}(:)) / sum(MSTref_AD(:)) * 100;     %for MSTref_AD = Occurmat_tot
            overlapPercentages_AD{k}(l) = overlapPercentage_AD{k};
        end
    end
    % Display mean and standard deviation
    % meanOverlap_AD = mean(overlapPercentages_AD{k});
    % stdDevOverlap_AD = std(overlapPercentages_AD{k});
    meanOverlap_AD(k) = sum(overlapPercentages_AD{k}) / (flag_AD(k));  % mean of
    dmean=overlapPercentages_AD{k}-meanOverlap_AD(k);
    num=flag_AD(k);
    dmean(num+1:numSubjects_AD)=0;
    stdDevOverlap_AD(k) = sqrt(sum((dmean).^2) / num);
    fprintf('Mean Overlap for AD: %.2f%%\n', meanOverlap_AD(k));
    fprintf('Standard Deviation for AD: %.2f%%\n', stdDevOverlap_AD(k));
end
OverallmeanOverlap_AD = mean(meanOverlap_AD)
OverallstdOverlap_AD = mean(stdDevOverlap_AD)

%% MST parameters for Ref
%% Diameter, Leaf fraction, Betweenness centrality, Degree,
Ref_AD = meanMST_AD_all;   %Senario #1
% Ref_AD = MSTmat_AD_all;    %Senario #2
Dist_AD = distance_wei(full(Ref_AD));
Diam_meanMatrix_AD = diameter(full(Ref_AD))
Deg_meanMatrix_AD = degrees_und(full(Ref_AD))
tree_meanMatrix_AD = graph(full(Ref_AD))
[Ec_AD, Rad, Diam, Cv, Pv] = grEccentricity(table2array(tree_meanMatrix_AD.Edges))
L_AD = numel(leaf_nodes(full(Ref_AD)))
M_AD = size(full(Ref_AD), 1)-1
LF_meanMatrix_AD = L_AD / M_AD
BC_meanMatrix_AD = betweenness_wei(full(Ref_AD))
BC_max_AD = max(BC_meanMatrix_AD)
Th_meanMatrix_AD = treeHierarchy(L_AD, M_AD, BC_max_AD)
k_AD = kappa(Deg_meanMatrix_AD)
%Identify Hub Nodes
meanDegreeAD_m = mean(Deg_meanMatrix_AD);
stdDegreeAD_m = std(Deg_meanMatrix_AD);
hubNodesDeg_AD_m = find(Deg_meanMatrix_AD > (meanDegreeAD_m + stdDegreeAD_m))  % Identify nodes with degree > mean + std
meanBetweennessAD_m = mean(BC_meanMatrix_AD);
stdBetweennessAD_m = std(BC_meanMatrix_AD);
hubNodesBC_AD_m = find(BC_meanMatrix_AD > (meanBetweennessAD_m + stdBetweennessAD_m))  % Identify nodes with betweenness centrality > mean + std

N=size(MSTmat_AD{1,1},1);
for k=1:size(MSTmat_AD,1)
    for t=1:size(MSTmat_AD,2)
        if ~isempty(MSTmat_AD{k,t})
            Diam_MSTmat_AD{k,t} = diameter(full(MSTmat_AD{k,t}));
            Deg_MSTmat_AD{k,t} = degrees_und(full(MSTmat_AD{k,t}));
            tree_MSTmat_AD{k,t} = graph(full(MSTmat_AD{k,t}));
            [Ec_MSTmat_AD{k,t}, Rad, Diam, Cv, Pv] = grEccentricity(table2array(tree_MSTmat_AD{k,t}.Edges));
            L_MSTmat_AD = numel(leaf_nodes(full(MSTmat_AD{k,t})));
            M_MSTmat_AD = size(full(MSTmat_AD{k,t}), 1)-1;
            LF_MSTmat_AD{k,t} = L_MSTmat_AD / M_MSTmat_AD;
            BC_MSTmat_AD{k,t} = (1/N^2)*betweenness_wei(full(MSTmat_AD{k,t}));
            BC_max_MSTmat_AD{k,t} = max(BC_MSTmat_AD{k,t});
            Th_MSTmat_AD{k,t} = treeHierarchy(L_MSTmat_AD, M_MSTmat_AD, BC_max_MSTmat_AD{k,t});
            k_MSTmat_AD{k,t} = 1./kappa(Deg_MSTmat_AD{k,t});
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Hub node definition (perplexity)
for k=1:size(MSTmat_AD,1)
    for t=1:size(MSTmat_AD,2)
        if ~isempty(MSTmat_AD{k,t})
            %Identify Hub Nodes
            meanDegree = mean(Deg_MSTmat_AD{k,t});
            stdDegree = std(Deg_MSTmat_AD{k,t});
            hubNodesDegree{k,t} = find(Deg_MSTmat_AD{k,t} > (meanDegree + stdDegree));  % Identify nodes with degree > mean + std
            meanBetweenness{k,t} = mean(BC_MSTmat_AD{k,t});
            stdBetweenness{k,t} = std(BC_MSTmat_AD{k,t});
            hubNodes_AD{k,t} = find(BC_MSTmat_AD{k,t} > (meanBetweenness{k,t} + stdBetweenness{k,t}));  % Identify nodes with betweenness centrality > mean + std
        end
    end
end

%% Similarity
%%%%%%%%%%%%%%%% between MSTref_AD of 36 subjects %%%%%%%%%%%%%%%%%%%%%%%%%

% MSTref_AD = double(MSTref_AD);
% for i=1:size(MSTmat_AD,1)
%     for j=1:size(MSTmat_AD,2)
%         if ~isempty(MSTmat_AD{i,j})
%             MSTmat_AD_bin{i,j} = double(logical(MSTmat_AD{i,j}));
%             Sim_edu{i,j} = pdist2(MSTref_AD, MSTmat_AD_bin{i,j}, 'euclidean'); %Euclidean distance
%             Sim_manh{i,j} = pdist2(MSTref_AD, MSTmat_AD_bin{i,j}, 'cityblock'); %Manhattan distance
%             % Sim_mahl = pdist2(MSTref_AD{1}, MSTref_AD{2}, 'mahalanobis', inv(cov(MSTref_AD{1}))); %Mahalanobis distance
%             Sim_cos{i,j} = 1 - dot(MSTref_AD, MSTmat_AD_bin{i,j}, 1); %/ (norm(double(MSTref_AD), 2) * norm(MSTmat_AD_bin{i,j}, 2)); %Cosine similarity
%             Sim_Jac(i,j) = sum(MSTref_AD & MSTmat_AD_bin{i,j}, 'all') / sum(MSTref_AD |MSTmat_AD_bin{i,j}, 'all'); %Jaccard similarity
%             Sim_Dic(i,j) = (2 * sum(MSTref_AD & MSTmat_AD_bin{i,j}, 'all')) / (sum(MSTref_AD, 'all') + sum(MSTmat_AD_bin{i,j}, 'all')); %Dice coefficient
%             % [Mreordered{i, j}, Mindices{i, j}, cost(i, j)] = align_matrices(MSTref_AD, MSTmat_AD_bin{i,j},'cosang',1);
%         end
%     end
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
WPLIAD_eeg_subj1 = zeros(size(WPLIAD_eeg2{1,1}));
flag=1;
n=5;
for t=1:size(WPLIAD_eeg2(n,:),2)
    if ~isempty(WPLIAD_eeg2{n,t})
        flag(t+1)=flag(t)+1;
        currentMat = WPLIAD_eeg2{n,t};
        WPLIAD_eeg_subj1 = WPLIAD_eeg_subj1 + currentMat;
    end
end
meanWPLIsubj1_AD = WPLIAD_eeg_subj1 / (numel(flag));
meanWPLIsubj1_AD = triu(meanWPLIsubj1_AD)+tril(transpose(meanWPLIsubj1_AD))-diag(diag(meanWPLIsubj1_AD));
figure, imagesc(meanWPLIsubj1_AD), title('WPLI of AD'),xlabel('Channels'),ylabel('Channels')
dlmwrite('meanWPLIsubj1_AD.txt', meanWPLIsubj1_AD, 'delimiter','\t')
figure, imagesc(meanMSTsubj_AD{n}), title('MST of AD subject'),xlabel('Channels'),ylabel('Channels')
dlmwrite('meanMSTsubj_AD.txt', meanMSTsubj_AD{n}, 'delimiter','\t')
