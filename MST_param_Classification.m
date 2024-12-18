% % %%
% % for i = 1:size(Diam_MSTmat_MCI, 1)
% %     for j = 1:size(Diam_MSTmat_MCI, 2)
% %         if Diam_MSTmat_MCI{i, j}==0;
% %             Diam_MSTmat_MCI{i, j} = [];
% %         end
% %     end
% % end
% % 
% % for i = 1:size(Diam_MSTmat_MCI, 1)
% %     for j = 1:size(Diam_MSTmat_MCI, 2)
% %         if ~isempty(Diam_MSTmat_MCI{i, j})
% %             Diam_MSTmat_MCI_m(i) = mean(cell2mat(Diam_MSTmat_MCI(i, :)));
% %         end
% %     end
% % end
% % 
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % for i = 1:size(Th_MSTmat_MCI, 1)
% %     for j = 1:size(Th_MSTmat_MCI, 2)
% %         if Th_MSTmat_MCI{i, j}==0;
% %             Th_MSTmat_MCI{i, j} = [];
% %         end
% %     end
% % end
% % 
% % for i = 1:size(Th_MSTmat_MCI, 1)
% %     for j = 1:size(Th_MSTmat_MCI, 2)
% %         if ~isempty(Th_MSTmat_MCI{i, j})
% %             Th_MSTmat_MCI_m(i) = mean(cell2mat(Th_MSTmat_MCI(i, :)));
% %         end
% %     end
% % end
% % 
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % for i = 1:size(k_MSTmat_MCI, 1)
% %     for j = 1:size(k_MSTmat_MCI, 2)
% %         if k_MSTmat_MCI{i, j}==0;
% %             k_MSTmat_MCI{i, j} = [];
% %         end
% %     end
% % end
% % 
% % for i = 1:size(k_MSTmat_MCI, 1)
% %     for j = 1:size(k_MSTmat_MCI, 2)
% %         if ~isempty(k_MSTmat_MCI{i, j})
% %             k_MSTmat_MCI_m(i) = mean(cell2mat(k_MSTmat_MCI(i, :)));
% %         end
% %     end
% % end
% % 
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % for i = 1:size(BC_max_MSTmat_MCI, 1)
% %     for j = 1:size(BC_max_MSTmat_MCI, 2)
% %         if BC_max_MSTmat_MCI{i, j}==0;
% %             BC_max_MSTmat_MCI{i, j} = [];
% %         end
% %     end
% % end
% % 
% % for i = 1:size(BC_max_MSTmat_MCI, 1)
% %     for j = 1:size(BC_max_MSTmat_MCI, 2)
% %         if ~isempty(BC_max_MSTmat_MCI{i, j})
% %             BC_max_MSTmat_MCI_m(i) = mean(cell2mat(BC_max_MSTmat_MCI(i, :)));
% %         end
% %     end
% % end
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % for i = 1:size(Ec_MSTmat_MCI, 1)
% %     for j = 1:size(Ec_MSTmat_MCI, 2)
% %         if Ec_MSTmat_MCI{i, j}==0
% %             Ec_MSTmat_MCI{i, j} = [];
% %         end
% %     end
% % end
% 
Ec_MSTmat_MCI_m={};
for i = 1:size(Ec_MSTmat_MCI, 1)
    for j = 1:size(Ec_MSTmat_MCI, 2)
        if ~isempty(Ec_MSTmat_MCI{i, j})
            Ec_MSTmat_MCI_m{i,j} = mean(Ec_MSTmat_MCI{i, j});
        end
    end
end
%
Ec_MSTmat_AD_m={};
for i = 1:size(Ec_MSTmat_AD, 1)
    for j = 1:size(Ec_MSTmat_AD, 2)
        if ~isempty(Ec_MSTmat_AD{i, j})
            Ec_MSTmat_AD_m{i,j} = mean(Ec_MSTmat_AD{i, j});
        end
    end
end
% 
Ec_MSTmat_HC_m = {};
for i = 1:size(Ec_MSTmat_HC, 1)
    for j = 1:size(Ec_MSTmat_HC, 2)
        if ~isempty(Ec_MSTmat_HC{i, j})
            Ec_MSTmat_HC_m{i,j} = mean(Ec_MSTmat_HC{i, j});
        end
    end
end
% 
% % for i = 1:size(Ec_MSTmat_MCI_m, 1)
% %     for j = 1:size(Ec_MSTmat_MCI_m, 2)
% %         if Ec_MSTmat_MCI_m{i, j}==0
% %             Ec_MSTmat_MCI_m{i, j} = [];
% %         end
% %     end
% % end
% % 
% % for i = 1:size(Ec_MSTmat_MCI_m, 1)
% %   Ec_MSTmat_MCI_m_m(i) = mean(cell2mat(Ec_MSTmat_MCI_m(i, :)));
% % end
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % for i = 1:size(Deg_MSTmat_MCI, 1)
% %     for j = 1:size(Deg_MSTmat_MCI, 2)
% %         if Deg_MSTmat_MCI{i, j}==0;
% %             Deg_MSTmat_MCI{i, j} = [];
% %         end
% %     end
% % end
Deg_MSTmat_AD_m={};
for i = 1:size(Deg_MSTmat_AD, 1)
    for j = 1:size(Deg_MSTmat_AD, 2)
        if ~isempty(Deg_MSTmat_AD{i, j})
            Deg_MSTmat_AD_m{i,j} = mean((Deg_MSTmat_AD{i, j}));
        end
    end
end
Deg_MSTmat_HC_m={};
for i = 1:size(Deg_MSTmat_HC, 1)
    for j = 1:size(Deg_MSTmat_HC, 2)
        if ~isempty(Deg_MSTmat_HC{i, j})
            Deg_MSTmat_HC_m{i,j} = mean((Deg_MSTmat_HC{i, j}));
        end
    end
end
Deg_MSTmat_MCI_m={};
for i = 1:size(Deg_MSTmat_MCI, 1)
    for j = 1:size(Deg_MSTmat_MCI, 2)
        if ~isempty(Deg_MSTmat_MCI{i, j})
            Deg_MSTmat_MCI_m{i,j} = mean((Deg_MSTmat_MCI{i, j}));
        end
    end
end
% 
% % for i = 1:size(Deg_MSTmat_MCI_m, 1)
% %     for j = 1:size(Deg_MSTmat_MCI_m, 2)
% %         if Deg_MSTmat_MCI_m{i, j}==0
% %             Deg_MSTmat_MCI_m{i, j} = [];
% %         end
% %     end
% % end
% % 
% % for i = 1:size(Deg_MSTmat_MCI_m, 1)
% %   Deg_MSTmat_MCI_m_m(i) = mean(cell2mat(Deg_MSTmat_MCI_m(i, :)));
% % end
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % for i = 1:size(LF_MSTmat_MCI, 1)
% %     for j = 1:size(LF_MSTmat_MCI, 2)
% %         if ~isempty(LF_MSTmat_MCI{i, j})
% %             LF_MSTmat_MCI_m(i) = mean(cell2mat(LF_MSTmat_MCI(i, :)));
% %         end
% %     end
% % end
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % for i = 1:size(LF_MSTmat_HC, 1)
% %     for j = 1:size(LF_MSTmat_HC, 2)
% %         if LF_MSTmat_HC{i, j}==0;
% %             LF_MSTmat_HC{i, j} = [];
% %         end
% %     end
% % end
% 
% % for i = 1:size(LF_MSTmat_HC, 1)
% %     for j = 1:size(LF_MSTmat_HC, 2)
% %         if ~isempty(LF_MSTmat_HC{i, j})
% %             LF_MSTmat_HC_m(i) = mean(cell2mat(LF_MSTmat_HC(i, :)));
% %         end
% %     end
% % end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% group1=[Th_MSTmat_AD_m'   Diam_MSTmat_AD_m'  LF_MSTmat_AD_m'   BC_max_MSTmat_AD_m'  Deg_MSTmat_AD_m_m'  k_MSTmat_AD_m'  Ec_MSTmat_AD_m_m'];
% group2=[Th_MSTmat_HC_m'   Diam_MSTmat_HC_m'  LF_MSTmat_HC_m'   BC_max_MSTmat_HC_m'  Deg_MSTmat_HC_m_m'  k_MSTmat_HC_m'  Ec_MSTmat_HC_m_m'];
% group3=[Th_MSTmat_MCI_m'   Diam_MSTmat_MCI_m'  LF_MSTmat_MCI_m'   BC_max_MSTmat_MCI_m'  Deg_MSTmat_MCI_m_m'  k_MSTmat_MCI_m'   Ec_MSTmat_MCI_m_m'];

% group1=[Th_MSTmat_AD   Diam_MSTmat_AD  LF_MSTmat_AD   BC_max_MSTmat_AD  Deg_MSTmat_AD_m  k_MSTmat_AD  Ec_MSTmat_AD_m];
% group2=[Th_MSTmat_HC   Diam_MSTmat_HC  LF_MSTmat_HC   BC_max_MSTmat_HC  Deg_MSTmat_HC_m  k_MSTmat_HC  Ec_MSTmat_HC_m];
% group3=[Th_MSTmat_MCI   Diam_MSTmat_MCI  LF_MSTmat_MCI   BC_max_MSTmat_MCI  Deg_MSTmat_MCI_m  k_MSTmat_MCI   Ec_MSTmat_MCI_m];

%%%%%%%%%% Omit Degree as a feature because it has equal mean value over
%%%%%%%%%% all channels %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
group1=[Th_MSTmat_AD   Diam_MSTmat_AD  LF_MSTmat_AD   BC_max_MSTmat_AD    k_MSTmat_AD    Ec_MSTmat_AD_m    Deg_MSTmat_AD_m];
group2=[Th_MSTmat_HC   Diam_MSTmat_HC  LF_MSTmat_HC   BC_max_MSTmat_HC    k_MSTmat_HC  Ec_MSTmat_HC_m  Deg_MSTmat_HC_m];
group3=[Th_MSTmat_MCI   Diam_MSTmat_MCI  LF_MSTmat_MCI   BC_max_MSTmat_MCI    k_MSTmat_MCI   Ec_MSTmat_MCI_m  Deg_MSTmat_MCI_m];
%
% group1=[Diam_MSTmat_AD  k_MSTmat_AD    Ec_MSTmat_AD_m];
% group2=[Diam_MSTmat_HC  k_MSTmat_HC  Ec_MSTmat_HC_m];
% group3=[Diam_MSTmat_MCI  k_MSTmat_MCI   Ec_MSTmat_MCI_m];
%
group2 = [group2 cell(size(group2,1), size(group1,2)-size(group2,2))];
group3 = [group3 cell(size(group3,1), size(group1,2)-size(group3,2))];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
group = [group1; group2; group3];
for i = 1:size(group, 1)
    for j = 1:size(group, 2)
        if isempty(group{i, j})
            group{i,j} = 0;
        end
    end
end
group = cell2mat(group);
y = [repmat({'AD'}, size(group1,1), 1); repmat({'HC'}, size(group2,1), 1); repmat({'MCI'}, size(group3,1), 1)];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% group = [group1; group2];
% for i = 1:size(group, 1)
%     for j = 1:size(group, 2)
%         if isempty(group{i, j})
%             group{i,j} = 0;
%         end
%     end
% end
% group = cell2mat(group);
% y = [repmat({'AD'}, size(group1,1), 1); repmat({'HC'}, size(group2,1), 1)];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% group = [group1; group3];
% for i = 1:size(group, 1)
%     for j = 1:size(group, 2)
%         if isempty(group{i, j})
%             group{i,j} = 0;
%         end
%     end
% end
% group = cell2mat(group);
% y = [repmat({'AD'}, size(group1,1), 1); repmat({'MCI'}, size(group3,1), 1)];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
group = [group2; group3];
for i = 1:size(group, 1)
    for j = 1:size(group, 2)
        if isempty(group{i, j})
            group{i,j} = 0;
        end
    end
end
group = cell2mat(group);
y = [repmat({'HC'}, size(group2,1), 1); repmat({'MCI'}, size(group3,1), 1)];

%%
num_features = size(group, 2);
p_values = zeros(num_features, 1);
for i = 1:num_features
    p = anova1(group(:, i), y, 'off');
    p_values(i) = p;
end
% Select significant features (e.g., p < 0.05)
significant_features = find(p_values < 0.05);
pval_significant = p_values(significant_features);
disp(['Number of significant features: ', num2str(length(significant_features))]);
% Extract significant features
X_significant = group(:, significant_features);
% Normalize the features
X_significant = normalize(X_significant);

y_categorical = categorical(y);

% Handle class imbalance using random oversampling
% X = group;
X = X_significant;
y = y_categorical;

% Convert y to numeric for resampling
y_numeric = grp2idx(y);

% Oversample the minority classes to balance the dataset
max_size = max([size(group1,1),size(group2,1),size(group3,1)]);
oversample_ratio = [max_size/size(group1,1), max_size/size(group2,1), max_size/size(group3,1)]; % AD is the majority class
[X_resampled, y_resampled] = resampleData(X, y_numeric, oversample_ratio);

% Convert resampled y back to categorical
y_resampled = categorical(y_resampled);

% PCA for dimensionality reduction
[coeff, score, ~, ~, explained] = pca(X_resampled);
% Select components that explain at least 95% of the variance
explained_variance = cumsum(explained);
num_components = find(explained_variance >= 95, 1);
X_reduced = score(:, 1:num_components);

%% CVpartition
% Split data into training and testing sets
cv = cvpartition(y_resampled, 'HoldOut', 0.3);
X_train = X_reduced(training(cv), :);
X_test = X_reduced(test(cv), :);
y_train = y_resampled(training(cv));
y_test = y_resampled(test(cv));

%% KNN
% k-NN Grid Search
% k = 2;
for k = 1:10
    % Mdl_knn = fitcknn(X_train, y_train, 'NumNeighbors', k);
    Mdl_knn = fitcknn(X_reduced, y_resampled, 'NumNeighbors', k);
    cv_knn = crossval(Mdl_knn, 'KFold', 10);
    accuracy_knn_cv = 1 - kfoldLoss(cv_knn)
    disp(['k-NN Test Accuracy: ', num2str(accuracy_knn_cv * 100), '%']);
end
% disp(['Best k-NN Accuracy: ', num2str(best_knn_accuracy * 100), '%, k: ', num2str(k)]);
Mdl_knn = fitcknn(X_train, y_train, 'NumNeighbors', k);
y_pred_knn = predict(Mdl_knn, X_test);
accuracy_knn = sum(y_pred_knn == y_test) / length(y_test);
% disp(['k-NN Test Accuracy: ', num2str(accuracy_knn * 100), '%']);

%% SVM
% SVM Grid Search
    % Mdl_svm = fitcecoc(X_train, y_train);
    % cv_svm = crossval(Mdl_svm, 'KFold', 10);
    % accuracy_svm_cv = 1 - kfoldLoss(cv_svm)
    % if accuracy_svm_cv > best_svm_accuracy
    %     best_svm_accuracy = accuracy_svm_cv;
    % end
% Mdl_svm = fitcsvm(X_reduced, y_resampled,'KernelFunction','rbf');
% cv_svm = crossval(Mdl_svm, 'KFold', 10);
% classLoss = kfoldLoss(cv_svm);
% accuracy_svm_cv = 1 - kfoldLoss(cv_svm)
% disp(['SVM Test Accuracy: ', num2str(accuracy_svm_cv * 100), '%']);

% Mdl_svm = fitcecoc(X_train, y_train, 'cvpartition', cv);
for i = 1:10
Mdl_svm = fitcecoc(X_reduced, y_resampled, 'cvpartition', cv);
% cv_svm = crossval(Mdl_svm, 'KFold', 10);
cv_svm = crossval(Mdl_svm, 'Holdout', 0.3);
classLoss = kfoldLoss(cv_svm);
accuracy_svm_cv = 1 - kfoldLoss(cv_svm);
end
% y_pred_svm = predict(Mdl_svm, X_test);
% accuracy_svm = sum(y_pred_svm == y_test) / length(y_test);
disp(['SVM Test Accuracy: ', num2str(accuracy_svm_cv * 100), '%']);

%% Decision Tree
% Decision Tree Grid Search
best_tree_accuracy = 0;
best_max_splits = 10;
cv = cvpartition(y_resampled, 'k', 10);
Mdl_tree = fitctree(X_reduced, y_resampled, 'CVPartition',cv);
err_tree = kfoldLoss(Mdl_tree);

% for max_splits = 2:10
    % Mdl_tree = fitctree(X_train, y_train, 'MaxNumSplits', max_splits)
    % cv_tree = crossval(Mdl_tree, 'KFold', 10)
    % accuracy_tree_cv = 1 - kfoldLoss(cv_tree);
%     if accuracy_tree_cv > best_tree_accuracy
%         best_tree_accuracy = accuracy_tree_cv;
%         best_max_splits = max_splits;
%     end
% end
% disp(['Best Decision Tree Accuracy: ', num2str(best_tree_accuracy * 100), '%, Best Max Splits: ', num2str(best_max_splits)]);
% Mdl_tree = fitctree(X_train, y_train, 'MaxNumSplits', best_max_splits);
% y_pred_tree = predict(Mdl_tree, X_test);
% accuracy_tree = sum(y_pred_tree == y_test) / length(y_test);
% disp(['Decision Tree Test Accuracy: ', num2str(accuracy_tree * 100), '%']);
disp(['Decision Tree Test Accuracy: ', num2str((1-err_tree) * 100), '%']);

%% Random Forest
% Mdl_rforest = TreeBagger(50,X_train, y_train,...
Mdl_rforest = TreeBagger(50,X_reduced, y_resampled,...
    Method="classification",...
    OOBPrediction="on")
view(Mdl_rforest.Trees{1},Mode="graph")

figure,plot(oobError(Mdl_rforest))
xlabel("Number of Grown Trees")
ylabel("Out-of-Bag Classification Error")

figure,plot(1-oobError(Mdl_rforest))
xlabel("Number of Grown Trees")
ylabel("Out-of-Bag Classification Accuracy")
disp(['Random Forest Test Accuracy: ', num2str(mean(1-oobError(Mdl_rforest)) * 100), '%']);

[v, z]=find(oobError(Mdl_rforest)==min(oobError(Mdl_rforest)));

oobLabels = oobPredict(Mdl_rforest);
ind = randsample(length(oobLabels),10);
% t=table(y_train(ind),oobLabels(ind),...
%     VariableNames=["TrueLabel" "PredictedLabel"]);
% 
% % cv_rforest = crossval(Mdl_rforest, 'KFold', 10)
% % accuracy_rforest_cv = 1 - kfoldLoss(cv_rforest);
% er = nnz(double(t.TrueLabel)-str2num(cell2mat(t.PredictedLabel)))/10;
% accuracy_rforest_cv = 1 - er;
% 
% % y_train = double(y_train);
% % t = templateTree('MaxNumSplits',5);
% % Mdl = fitcensemble(X_train, y_train,'Method','AdaBoostM1','Learners',t,'CrossVal','on');
% % kflc = kfoldLoss(Mdl,'Mode','cumulative');
% % figure;
% % plot(kflc);
% % ylabel('10-fold Misclassification rate');
% % xlabel('Learning cycle');
% % estGenError = kflc(end)
% 
% % Define cross-validation parameters (e.g., 10-fold)
% cvp = cvpartition(size(X_test,1), 'KFold', 10);  % Adjust KFold value for different folds
% % Perform cross-validation with the model and data
% % y_pred_rforest = predict(model, X_test);
% % accuracy_rforest = sum(y_pred_rforest == y_test) / length(y_test);
% 
% %% Split data into folds (e.g., 10-fold)
% % cp = round(linspace(1, size(data,1), 10));
% % for i = 1:10
% %     % Define training and testing sets for each fold
% %     train_idx = cp(i:end-1);
% %     test_idx = cp(i+1);
% %     X_train = X_reduced(train_idx, 1:end-1);
% %     y_train = y_resampled(train_idx, end);
% %     X_test = X_reduced(test_idx, 1:end-1);
% %     y_test = y_resampled(test_idx, end);
% % 
% %     % Train the Random Forest model and evaluate on the test set (repeat for each fold)
% %     model = fitcensemble(X_train, y_train);  % Train the model
% %     y_pred = predict(model, X_test);       % Make predictions on test set
% %     % Evaluate performance using metrics (accuracy, precision, etc.)
% %     accuracy_rforest = sum(y_pred_rforest == y_test) / length(y_test)
% % end
% 
