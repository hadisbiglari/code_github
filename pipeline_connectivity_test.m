% Time estimates for 60 epochs at 100 Hz in second
% Cross-spectrum                    2
% Coherence                         2
% Weighted Phase Lag Index          49
% Granger Causality (GC)            619
% Time-reversed GC                  -
% Partial Directed Coherence (PDC)  152
% Time-reversed PDC                 -
% Directed Transfer Entropy (DTF)   167
% Time-reversed DTF                 -
% Multivariate Interaction Measure  19
% Maximized Imaginary Coherency     18 

clear
eeglab

eeglabp = fileparts(which('eeglab.m'));
% EEG = pop_loadset('filename','eeglab_data_epochs_ica.set','filepath',fullfile(eeglabp, 'sample_data/'));
EEG = pop_loadset('filename','sub-008_task-eyesclosed_eeg.set','filepath',fullfile('C:\Users\MIQDAD\Desktop\PhD\PROPOSED Method\EEG_AD\OpenNeuroDatasets\ds004504-main\ds004504-main\derivatives\sub-008'));     % for my AD EEG
EEG = pop_resample( EEG, 100);
% EEG = pop_epoch( EEG, { }, [-0.5 1.5], 'newname', 'EEG Data epochs epochs', 'epochinfo', 'yes'); %
% EEG = pop_select( EEG, 'trial',1:30);
% EEG = pop_select( EEG, 'point', [1 299900]);  % for my AD EEG
% EEG = pop_select( EEG, 'trial', 1:round(EEG.pnts/10));
[ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG);
eeglab redraw;

EEG = pop_dipfit_settings( EEG, 'hdmfile',fullfile(eeglabp, 'plugins','dipfit','standard_BEM','standard_vol.mat'), ...
    'coordformat','MNI','mrifile',fullfile(eeglabp, 'plugins','dipfit','standard_BEM','standard_mri.mat'),...
    'chanfile',fullfile(eeglabp, 'plugins','dipfit','standard_BEM','elec', 'standard_1005.elc'),...
    'coord_transform',[0.83215 -15.6287 2.4114 0.081214 0.00093739 -1.5732 1.1742 1.0601 1.1485] );

EEG = pop_leadfield(EEG, 'sourcemodel',fullfile(eeglabp,'plugins','dipfit','LORETA-Talairach-BAs.mat'), ...
    'sourcemodel2mni',[0 -24 -45 0 0 -1.5708 1000 1000 1000] ,'downsample',1);

% EEG = pop_leadfield(EEG, 'sourcemodel',fullfile(eeglabp,'functions','supportfiles','head_modelColin27_5003_Standard-10-5-Cap339.mat'), ...
    % 'sourcemodel2mni',[0 -24 -45 0 0 -1.5708 1000 1000 1000] ,'downsample',1);

% EEG = pop_leadfield(EEG, 'sourcemodel',fullfile(eeglabp,'functions','supportfiles','colin27headmesh.mat'), ...
%     'sourcemodel2mni',[0 -24 -45 0 0 -1.5708 1000 1000 1000] ,'downsample',1);  % for my AD EEG

% EEG = pop_roi_activity(EEG, 'leadfield',EEG.dipfit.sourcemodel,'model','LCMV','modelparams',{0.05},'atlas','LORETA-Talairach-BAs','nPCA',3, 'chansel', EEG.dipfit.chansel);
EEG = pop_roi_activity(EEG, 'leadfield',EEG.dipfit.sourcemodel,'model','LCMV','modelparams',{0.05},'atlas','LORETA-Talairach-BAs','nPCA',3);
% EEG = pop_roi_activity(EEG, 'leadfield',EEG.dipfit.sourcemodel,'model','LCMV','modelparams',{0.05},'atlas','LORETA-Talairach-BAs','nPCA',3);  % for my AD EEG

% measures = { 'CS' 'aCOH' 'DTF'  'wPLI'  'PDC'  'MIM'  'MIC' 'GC' };  %Edit 'COH' to 'aCOH' by Hadis
% measures = { 'CS' 'COH' 'wPLI'  'PDC'};
% measures = { 'CS' 'COH' 'MIM' 'GC' };
% measures = { 'CS' 'COH' 'MIM' };
% measures = { 'TRGC' 'MIM' };
% measures = { 'MIM' 'iCOH'};
measures = { 'wPLI' };

for iMeasure = 1:length(measures)
    tic
    EEG = pop_roi_connect(EEG, 'methods', measures(iMeasure));
    t(iMeasure) = toc;
end
% pop_roi_connectplot(EEG, 'measure', 'MIM', 'plotcortex', 'on', 'plotmatrix', 'on', 'plotbutterfly', 'on');  
% pop_roi_connectplot(EEG, 'measure', 'MIM', 'plotcortex', 'on', 'plotmatrix', 'on');  %Edit by Hadis
% pop_roi_connectplot(EEG, 'measure', 'MIM', 'plot3d', 'on',  'plotcortex', 'off', 'plot3dparams', {'thresholdper', 0.05, 'brainmovieopt' {'nodeColorDataRange', [], 'nodeSizeLimits', [0 0.2]}});

% A = EEG.roi.GC(1, :, :);
% A = EEG.roi.wPLI(2, :, :);
% A = EEG.roi.aCOH(1, :, :);
A = EEG.roi.DTF(1, :, :);
[K,I] = min(abs(A),[],3);
M=68;
N=68;
B = zeros(M,N);

for i = 1:M
    for j = 1:N
        B(i,j) = A(1, i,j);
    end
end
figure, imagesc(B), title('DTF')

A=EEG.roi.wPLI(:, :, 1);
for i = 1:M
    for j = 1:N
        B(i,j) = A(i, j, 1);
    end
end
figure, imagesc(B), title('wPLI')
