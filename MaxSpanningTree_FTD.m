%% MCI
% clear;
clc
% close all;
addpath('C:/Users/MIQDAD/Desktop/PhD/PROPOSED Method/Graph analysis_MST/Code');
addpath('C:/Users/MIQDAD/Desktop/PhD/PROPOSED Method/Graph analysis_MST/Code/2019_03_03_BCT')
addpath('C:/Users/MIQDAD/Desktop/PhD/PROPOSED Method/Graph analysis_MST/Code/FCLAB-master/FCLAB-master/FCLAB1.0.0/FC_metrics')
addpath('C:/Users/MIQDAD/Desktop/PhD/PROPOSED Method/Graph analysis_MST/Code/FCLAB-master/FCLAB-master/FCLAB1.0.0/MST_params')
load('EEGFD.mat');
% load('PLIFD_eeg');
% ConnMCI_all = PLIFD_eeg;
% load('MEGMCI.mat');
% load('PLIMCI');
% load('plvMCI_all.mat')
% load('WPLIMCI2.mat')
load('WPLIFD_eeg2');
% ConnMCI_all = WPLIMCI2;
ConnFTD_all = WPLIFD_eeg2;
% WPLIMCI2_M = WPLIMCI2([1,2,3,4,5,6,8,11,14,16],:); % subject 2,3,4,6,8,11,23 resultsa huge difference with female group and pluse-like lines in static graph 
% WPLIMCI2_F = WPLIMCI2([7,9,10,12,13,15,17,18],:);
% ConnMCI_all = WPLIMCI2_F;
% ConnMCI_all = WPLIMCI2_M;
% ConnMCI_all = [WPLIMCI2_F; WPLIMCI2_M];
% ConnMCI_all = WPLIFD_eeg2;
% WPLIFD_eeg2_M = WPLIFD_eeg2([1,2,3,4,6,7,11,12,13,17,20,21,22,23]);
% WPLIFD_eeg2_F = WPLIFD_eeg2([5,8,9,10,14,15,16,18,19]);
% ConnMCI_all = WPLIFD_eeg2_F;
% fs = 678.19;
% longest data = 27x1695xfs = 67.48 sec
% shortest data = 4x1695xfs = 10 sec
%% Find Static graph for MCI
%% Over each subjest across time
%%%%%%%%%%% Graph of all frames of all subjects %%%%%%%%%%%%%%%
MSTmat_FTD={};
for k=1:size(ConnFTD_all,1)
    for t=1:size(ConnFTD_all,2)
        if ~isempty(ConnFTD_all{k,t})
            % figure, imagesc(ConnMCI_all{k,t}), title('Connectivity matrix')
            S_FTD = [];
            t_FTD=[];
            for i = 1:size(ConnFTD_all{1}, 1)
                start_node = i*ones(size(ConnFTD_all{1}, 1)-i, 1);
                end_node = i+1:size(ConnFTD_all{1}, 1);
                % Append the start and end nodes to the S and t vectors
                S_FTD = [S_FTD start_node'];
                t_FTD = [t_FTD end_node];
            end
            for j=1:size(S_FTD,2)
                weights_FTD_inv_all{k,t}(j)=1./(ConnFTD_all{k,t}(S_FTD(j),t_FTD(j)));
            end
            % Convert the matrix to a graph object
            G_FTD = graph(S_FTD,t_FTD,weights_FTD_inv_all{k, t});
            G_FTD.Edges;
            k;
            tree_FTD{k, t} = minspantree(G_FTD);
            tree_FTD{k, t}.Edges;
            weights_FTD = tree_FTD{k}.Edges.Weight;
            weights_FTD = (weights_FTD).^-1;
            % figure, plot(G_MCI), title('MST graph of MCI')
            % highlight(p,tree_HC{k})
            MSTmat_FTD{k, t} = adjacency(tree_FTD{k, t}, double(weights_FTD));    %Convert MST graph to MST matrix
            % figure, imagesc(MSTmat_MCI{k}), title('MST matrix of MCI')
        end
    end
end

%% Senarios for static graph calculation:
%% Senario 1: mean of MSTmat_HC over each subject (t) and then over all of subjects (k)
meanMSTsubj_FTD = {};
MSTmat_FTD_all = cell(size(MSTmat_FTD,1),1);
flag_FTD=zeros(size(MSTmat_FTD,1),1);
for k=1:size(MSTmat_FTD,1)
    MSTmat_FTD_all{k} = zeros(size(MSTmat_FTD{1}));
    for t=1:size(MSTmat_FTD,2)
        if ~isempty(MSTmat_FTD{k,t})
            flag_FTD(k)=flag_FTD(k)+1;
            currentMatrix_FTD_all = MSTmat_FTD{k,t};
            MSTmat_FTD_all{k} = MSTmat_FTD_all{k} + currentMatrix_FTD_all;
        end
    end
    meanMSTsubj_FTD{k,1} = MSTmat_FTD_all{k,1} / (flag_FTD(k));
    % figure,imagesc(meanMSTmat_MCI_all{k}),title('meanMST MCI for each subject');
end
for k=1:size(meanMSTsubj_FTD,1)
    curr = meanMSTsubj_FTD{k};
    sumMST_FTD_all = meanMSTsubj_FTD{k} + curr;
end
meanMST_FTD_all = sumMST_FTD_all/size(meanMSTsubj_FTD,1);
% Normalize elements using min-max scaling
min_val = min(meanMST_FTD_all(:));
max_val = max(meanMST_FTD_all(:));
meanMST_FTD_all = (meanMST_FTD_all - min_val) ./ (max_val - min_val);

thr = max(max(meanMST_FTD_all))*0.25;
meanMST_FTD_all_thr = meanMST_FTD_all.*(meanMST_FTD_all>thr);
figure, imagesc(meanMST_FTD_all), title('Disease specific matrix of FTD'),xlabel('Channels'),ylabel('Channels')
figure, imagesc(meanMST_FTD_all_thr), title('Disease specific matrix of FTD after thresholding')
dlmwrite('Staticgrah_FTD.txt', meanMST_FTD_all, 'delimiter','\t')
dlmwrite('Staticgrah_FTD_thr.txt', meanMST_FTD_all_thr, 'delimiter','\t')

%% Senario 2: mean of connectivity matrices over t and then k, then calculate MST for it
%%%%%%%%%%% MST of mean of Occurence matrices (This one is a better refernce for static subgraph calculation) %%%%%%%%%%%%%%
%%%% Ocuurence matrix is defined as summation of all Connectivities for each subject
MSTMCI_mean_all = {};
MSTMCI_meann_all = {};
meanConn_MCI = {};
OccurMatrix_MCI_all = cell(size(ConnFTD_all,1),1);
for k=1:size(ConnFTD_all,1)
    OccurMatrix_MCI_all{k} = zeros(size(ConnFTD_all{1}));
    for t=1:size(ConnFTD_all,2)
        if ~isempty(ConnFTD_all{k,t})
            currentMatrix_FTD_all = ConnFTD_all{k,t};
            OccurMatrix_MCI_all{k} = OccurMatrix_MCI_all{k} + currentMatrix_FTD_all;
        end
    end
    meanConn_MCI{k,1} = OccurMatrix_MCI_all{k,1} / (flag_FTD(k));  % Assuming you have 36 matrices in the cell array
    % figure,imagesc(meanMatrix_MCI_all{k}),title('meanMatrix MCI for each subject');
end

for k=1:size(meanConn_MCI,1)
    curre = meanConn_MCI{k};
    sumConn_MCI_all = meanConn_MCI{k} + curre;
end
meanConn_MCI_all = sumConn_MCI_all/size(meanConn_MCI,1);
% MST for mean connectivity over time (t)
for k=1:size(meanConn_MCI,1)
    for j = 1:size(S_FTD,2)
        weightsmean_MCI_inv{k}(j)=1./(meanConn_MCI{k}(S_FTD(j),t_FTD(j)));
    end
    Gmean_MCI = graph(S_FTD,t_FTD,weightsmean_MCI_inv{k});
    Gmean_MCI.Edges;
    treemean_MCI{k} = minspantree(Gmean_MCI);
    treemean_MCI{k}.Edges;
    for i = 1:size(meanConn_MCI_all, 1)-1
        treemean_MCI_nod = treemean_MCI{k}.Edges.EndNodes;
        a = treemean_MCI_nod(:,1);
        b = treemean_MCI_nod(:,2);
        treemean_MCI_wht = treemean_MCI{k}.Edges.Weight;
        MSTMCI_mean_all{k}(a(i),b(i)) = 1/treemean_MCI_wht(i);
    end
    MSTMCI_mean_all{k} = padarray(MSTMCI_mean_all{k},[size(ConnFTD_all{1}, 1)-size(MSTMCI_mean_all{k},1) 0],0,'post');
    MSTMCI_meann_all{k} = triu(MSTMCI_mean_all{k})+tril(transpose(MSTMCI_mean_all{k}));
end
% MST for mean connectivity over time and then over subjects
for j = 1:size(S_FTD,2)
    weightsmean_MCI_inv_all(j)=1./(meanConn_MCI_all(S_FTD(j),t_FTD(j)));
end
Gmean_MCI_all = graph(S_FTD,t_FTD,weightsmean_MCI_inv_all);
Gmean_MCI_all.Edges;
treemean_MCI_all = minspantree(Gmean_MCI_all);
treemean_MCI_all.Edges;
weights_MCI_all = treemean_MCI_all.Edges.Weight;
weights_MCI_all = (weights_MCI_all).^-1;
% figure, plot(Gmean_MCI_all), title('MST graph of mean of all connectivities for MCI')
% highlight(p,treemean_HC_all)
MSTmat_FTD_all = adjacency(treemean_MCI_all, double(weights_MCI_all));    %Convert MST graph to MST matrix
thr2 = max(max(full(MSTmat_FTD_all)))*0.25;
MSTmat_MCI_all_thr = MSTmat_FTD_all.*(MSTmat_FTD_all>thr2);
figure, imagesc(MSTmat_FTD_all), title('MST matrix of mean of all connectivities for MCI')
figure, imagesc(MSTmat_MCI_all_thr), title('MST matrix of mean of all connectivities for MCI after threshold')
% dlmwrite('MSTmat_MCI_all.txt', MSTmat_MCI_all, 'delimiter','\t')

%% Overlap Percentage between mean MST matrix and each MST over time for each Subject %%%%%%%%%%%%%
numSubjects_MCI = size(MSTmat_FTD, 2);
% MSTsubjects_MCI = reshape(cell2mat(MSTMCI_all(:)).',148,148,numSubjects_MCI);
MSTref_MCI = logical(MSTmat_FTD_all);
for k=1:size(MSTMCI_meann_all,2)
    % MSTref_MCI{k} = MSTMCI_meann_all{k}; % one ref for each subject
    overlapPercentages_MCI{k} = zeros(1, numSubjects_MCI);
    for l = 1:numSubjects_MCI
        if ~isempty(cell2mat(MSTmat_FTD(k,l)))
            MSTsubjects_MCI{k,l} = logical(MSTmat_FTD{k,l});
            % overlapMatrix_MCI{k} = MSTref_MCI{k} & MSTsubjects_MCI(:, :, i);
            overlapMatrix_MCI{k} = MSTref_MCI & MSTsubjects_MCI{k,l};        % one ref for each subject
            % overlapMatrix_MCI{k,l} = MSTref_MCI & cell2mat(MSTsubjects_MCI(k,l));      %for MSTref_MCI = Occurmat_tot
            overlapPercentage_MCI{k} = sum(full(overlapMatrix_MCI{k}(:))) / sum(full(MSTref_MCI(:))) * 100;         % one ref for each subject
            % overlapPercentage_MCI{k,l} = sum(overlapMatrix_MCI{k}(:)) / sum(MSTref_MCI(:)) * 100;     %for MSTref_AD = Occurmat_tot
            overlapPercentages_MCI{k}(l) = overlapPercentage_MCI{k};
        end
    end
    % Display mean and standard deviation
    % meanOverlap_MCI = mean(overlapPercentages_MCI{k});
    % stdDevOverlap_MCI = std(overlapPercentages_MCI{k});
    meanOverlap_MCI(k) = sum(overlapPercentages_MCI{k}) / (flag_FTD(k));  % mean of
    dmean=overlapPercentages_MCI{k}-meanOverlap_MCI(k);
    num=flag_FTD(k);
    dmean(num+1:numSubjects_MCI)=0;
    stdDevOverlap_MCI(k) = sqrt(sum((dmean).^2) / num);
    fprintf('Mean Overlap for MCI: %.2f%%\n', meanOverlap_MCI(k));
    fprintf('Standard Deviation for MCI: %.2f%%\n', stdDevOverlap_MCI(k));
end
OverallmeanOverlap_MCI = mean(meanOverlap_MCI)
OverallstdOverlap_MCI = mean(stdDevOverlap_MCI)

%% MST parameters for Ref
%% Diameter, Leaf fraction, Betweenness centrality, Degree,
Ref_MCI = meanMST_FTD_all;   %Senario #1
% Ref_MCI = MSTmat_MCI_all;    %Senario #2
Dist_MCI = distance_wei(full(Ref_MCI));
Diam_meanMatrix_MCI = diameter(full(Ref_MCI))
Deg_meanMatrix_MCI = degrees_und(full(Ref_MCI))
tree_meanMatrix_MCI = graph(full(Ref_MCI))
[Ec_MCI, Rad, Diam, Cv, Pv] = grEccentricity(table2array(tree_meanMatrix_MCI.Edges))
L_MCI = numel(leaf_nodes(full(Ref_MCI)))
M_MCI = size(full(Ref_MCI), 1)-1
LF_meanMatrix_MCI = L_MCI / M_MCI
BC_meanMatrix_MCI = betweenness_wei(full(Ref_MCI))
BC_max_MCI = max(BC_meanMatrix_MCI)
Th_meanMatrix_MCI = treeHierarchy(L_MCI, M_MCI, BC_max_MCI)
k_MCI = kappa(Deg_meanMatrix_MCI)
%Identify Hub Nodes
meanDegreeMCI_m = mean(Deg_meanMatrix_MCI);
stdDegreeMCI_m = std(Deg_meanMatrix_MCI);
hubNodesDeg_MCI_m = find(Deg_meanMatrix_MCI > (meanDegreeMCI_m + stdDegreeMCI_m))  % Identify nodes with degree > mean + std
meanBetweennessMCI_m = mean(BC_meanMatrix_MCI);
stdBetweennessMCI_m = std(BC_meanMatrix_MCI);
hubNodesBC_MCI_m = find(BC_meanMatrix_MCI > (meanBetweennessMCI_m + stdBetweennessMCI_m))  % Identify nodes with betweenness centrality > mean + std

N=size(MSTmat_FTD{1,1},1);
for k=1:size(MSTmat_FTD,1)
    for t=1:size(MSTmat_FTD,2)
        if ~isempty(MSTmat_FTD{k,t})
            Diam_MSTmat_MCI{k,t} = diameter(full(MSTmat_FTD{k,t}));
            Deg_MSTmat_MCI{k,t} = degrees_und(full(MSTmat_FTD{k,t}));
            tree_MSTmat_MCI{k,t} = graph(full(MSTmat_FTD{k,t}));
            [Ec_MSTmat_MCI{k,t}, Rad, Diam, Cv, Pv] = grEccentricity(table2array(tree_MSTmat_MCI{k,t}.Edges));
            L_MSTmat_MCI = numel(leaf_nodes(full(MSTmat_FTD{k,t})));
            M_MSTmat_MCI = size(full(MSTmat_FTD{k,t}), 1)-1;
            LF_MSTmat_MCI{k,t} = L_MSTmat_MCI / M_MSTmat_MCI;
            BC_MSTmat_MCI{k,t} = (1/N^2)*betweenness_wei(full(MSTmat_FTD{k,t}));
            BC_max_MSTmat_MCI{k,t} = max(BC_MSTmat_MCI{k,t});
            Th_MSTmat_MCI{k,t} = treeHierarchy(L_MSTmat_MCI, M_MSTmat_MCI, BC_max_MSTmat_MCI{k,t});
            k_MSTmat_MCI{k,t} = 1./kappa(Deg_MSTmat_MCI{k,t});
        end
    end
end

%% Similarity
%%%%%%%%%%%%%%%% between MSTref_MCI of n subjects %%%%%%%%%%%%%%%%%%%%%%%%%
% MSTref_MCI = double(MSTref_MCI);
% for i=1:size(MSTmat_MCI,1)
%     for j=1:size(MSTmat_MCI,2)
%         if ~isempty(MSTmat_MCI{i,j})
%             MSTmat_MCI_bin{i,j} = double(logical(MSTmat_MCI{i,j}));
%         Sim_edu_MCI{i,j} = pdist2(MSTref_MCI, MSTmat_MCI_bin{i,j}, 'euclidean'); %Euclidean distance
%         Sim_manh_MCI{i,j} = pdist2(MSTref_MCI, MSTmat_MCI_bin{i,j}, 'cityblock'); %Manhattan distance
%         % Sim_mahl_MCI = pdist2(MSTref_MCI{1}, MSTref_MCI{2}, 'mahalanobis', inv(cov(MSTref_MCI{1}))); %Mahalanobis distance
%         Sim_cos_MCI{i,j} = 1 - dot(MSTref_MCI, MSTmat_MCI_bin{i,j}, 2); %/ (norm(MSTref_MCI, 2) * norm(MSTmat_MCI_bin{i,j}, 2)); %Cosine similarity
%         Sim_Jac_MCI(i,j) = sum(MSTref_MCI & MSTmat_MCI_bin{i,j}, 'all') / sum(MSTref_MCI |MSTmat_MCI_bin{i,j}, 'all'); %Jaccard similarity
%         Sim_Dic_MCI(i,j) = (2 * sum(MSTref_MCI & MSTmat_MCI_bin{i,j}, 'all')) / (sum(MSTref_MCI, 'all') + sum(MSTmat_MCI_bin{i,j}, 'all')); %Dice coefficient
%         % [Mreordered_MCI{i, j}, Mindices_MCI{i, j}, cost_MCI(i, j)] = align_matrices(MSTref_MCI{i} & MSTref_MCI{j},'cosang',1);
%         end
%     end
% end

for i=1:size(MSTmat_FTD,1)
    for j=1:size(MSTmat_FTD,2)-1
        if ~isempty(MSTmat_FTD{i,j})
            MSTmat_MCI_bin{i,j} = double(logical(MSTmat_FTD{i,j}));
            MSTmat_MCI_bin{i,j+1} = double(logical(MSTmat_FTD{i,j+1}));
        Sim_edu_MCI{i,j} = pdist2(MSTmat_MCI_bin{1,1}, MSTmat_MCI_bin{i,j+1}, 'euclidean'); %Euclidean distance
        Sim_manh_MCI{i,j} = pdist2(MSTmat_MCI_bin{i,j}, MSTmat_MCI_bin{i,j+1}, 'cityblock'); %Manhattan distance
        % Sim_mahl_MCI = pdist2(MSTref_MCI{1}, MSTref_MCI{2}, 'mahalanobis', inv(cov(MSTref_MCI{1}))); %Mahalanobis distance
        Sim_cos_MCI{i,j} = 1 - dot(MSTmat_MCI_bin{1,1}, MSTmat_MCI_bin{i,j+1}, 2); %/ (norm(MSTmat_MCI_bin{1,1}, 2) * norm(MSTmat_MCI_bin{i,j+1}, 2)); %Cosine similarity
        Sim_Jac_MCI(i,j) = sum(MSTmat_MCI_bin{i,j} & MSTmat_MCI_bin{i,j+1}, 'all') / sum(MSTmat_MCI_bin{i,j} |MSTmat_MCI_bin{i,j+1}, 'all'); %Jaccard similarity
        Sim_Dic_MCI(i,j) = (2 * sum(MSTmat_MCI_bin{i,j} & MSTmat_MCI_bin{i,j+1}, 'all')) / (sum(MSTmat_MCI_bin{i,j}, 'all') + sum(MSTmat_MCI_bin{i,j+1}, 'all')); %Dice coefficient
        % [Mreordered_MCI{i, j}, Mindices_MCI{i, j}, cost_MCI(i, j)] = align_matrices(MSTref_MCI{i} & MSTref_MCI{j},'cosang',1);
        i
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% اینجا اومدی از WPLI  استفاده کردی نه از MST...برای همین نتایج متفاوت شده!!!
WPLIFD_eeg2_F = WPLIMCI2_F;
meanMSTfemale_MCI = {};
flagg_MCI=zeros(size(WPLIFD_eeg2_F,1),1);
for k=1:size(WPLIFD_eeg2_F,1)
    WPLIMCI2_F_all{k} = zeros(size(WPLIFD_eeg2_F{1}));
    for t=1:size(WPLIFD_eeg2_F,2)
        if ~isempty(WPLIFD_eeg2_F{k,t})
            flagg_MCI(k)=flagg_MCI(k)+1;
            currentMatrix_WPLIMCI2_F_all = WPLIFD_eeg2_F{k,t};
            WPLIMCI2_F_all{k} = WPLIMCI2_F_all{k} + currentMatrix_WPLIMCI2_F_all;
        end
    end
    meanMSTfemale_MCI{k} = WPLIMCI2_F_all{k} / (flagg_MCI(k));
    % figure,imagesc(meanMSTmat_MCI_all{k}),title('meanMST MCI for each subject');
end
for k=1:size(meanMSTfemale_MCI,1)
    curr = meanMSTfemale_MCI{k};
    sumMST_MCI_female = meanMSTfemale_MCI{k} + curr;
end
meanMST_MCI_female = sumMST_MCI_female/size(meanMST_MCI_female,1);
mn_val = min(meanMST_MCI_female(:));
mx_val = max(meanMST_MCI_female(:));
meanMST_MCI_female = (meanMST_MCI_female - mn_val) ./ (mx_val - mn_val);
figure, imagesc(meanMST_MCI_female), title('mean MST matrix of female MCI subjects')
figure, imagesc(meanMST_MCI_male), title('mean MST matrix of male MCI subjects')
meanMST_MCI_malefemale = meanMST_MCI_male + meanMST_MCI_female /2;
figure, imagesc(ans), title('mean MST matrix of MCI')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Consistency of graphs in time
for k = 1:size(MSTmat_FTD(13,:),2)
    if ~isempty(MSTmat_FTD{13,k})
    figure, imagesc(MSTmat_FTD{13,k})
    end
end

