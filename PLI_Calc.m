% fs = 678.19;
Path1 = 'C:/Users/MIQDAD/Desktop/PhD/MEG_AD/MEG_50863_noECG_10s/ALZ';
FileList_AD = dir(fullfile(Path1));
for iFile_AD = 3:numel(FileList_AD)
    File_AD = fullfile(FileList_AD(iFile_AD).folder, FileList_AD(iFile_AD).name);
    FileList2 = dir(fullfile(File_AD));
    for k = 3:numel(FileList2)
        File_AD = fullfile(FileList2(k).folder, FileList2(k).name);
        MEG_AD{iFile_AD-2, k-2} = load(File_AD);
        MEGAD{iFile_AD-2, k-2} = MEG_AD{iFile_AD-2, k-2}.meg_no_ecg;
    end
end
%
%
Path2 = 'C:/Users/MIQDAD/Desktop/PhD/MEG_AD/MEG_50863_noECG_10s/CONT';
FileList_HC = dir(fullfile(Path2));
for iFile_HC = 3:numel(FileList_HC)
    File_HC = fullfile(FileList_HC(iFile_HC).folder, FileList_HC(iFile_HC).name);
    FileList2 = dir(fullfile(File_HC));
    for k = 3:numel(FileList2)
        File_HC = fullfile(FileList2(k).folder, FileList2(k).name);
        MEG_HC{iFile_HC-2, k-2} = load(File_HC);
        MEGHC{iFile_HC-2, k-2} = MEG_HC{iFile_HC-2, k-2}.meg_no_ecg;
    end
end

Path3 = 'C:/Users/MIQDAD/Desktop/PhD/MEG_ad/MEG_50863_noECG_10s/MCI';
FileList_MCI = dir(fullfile(Path3));
for iFile_MCI = 3:numel(FileList_MCI)
    File_MCI = fullfile(FileList_MCI(iFile_MCI).folder, FileList_MCI(iFile_MCI).name);
    FileList2 = dir(fullfile(File_MCI));
    for k = 3:numel(FileList2)
        File_MCI = fullfile(FileList2(k).folder, FileList2(k).name);
        MEG_MCI{iFile_MCI-2, k-2} = load(File_MCI);
        MEGMCI{iFile_MCI-2, k-2} = MEG_MCI{iFile_MCI-2, k-2}.meg_no_ecg;
    end
end
%
%
% %% PLI Calculation for MEG
PLIAD = {};
for i = 1:size(MEGAD, 1)
    for t = 1:size(MEGAD(1,:), 2)
        if ~isempty(MEGAD{i,t})
            % for j = 1:size(MEGAD{i,t}, 1)
            %     for k = j:size(MEGAD{i,t}, 1)
            %         PLIAD{i,t}(j, k) = PLI(MEGAD{i,t}(j,:), MEGAD{i,t}(k,:));
            %     end
            % end
            PLIAD{i,t} = PLI(MEGAD{i,t}, MEGAD{i,t});
        end
        t
    end
    i
end
%
%
PLIHC = {};
for i = 1:size(MEGHC, 1)
    for t = 1:size(MEGHC(1,:), 2)
        if ~isempty(MEGHC{i,t})
        %     for j = 1:size(MEGHC{i,t}, 1)
        %         for k = j:size(MEGHC{i,t}, 1)
        %             PLIHC{i,t}(j, k) = PLI(MEGHC{i,t}(j,:), MEGHC{i,t}(k,:));
        %         end
        %     end
          PLIHC{i,t} = PLI(MEGHC{i,t}, MEGHC{i,t});
        end
        t
    end
    i
end
%
%
PLIMCI = {};
for i = 1:size(MEGMCI, 1)
    for t = 1:size(MEGMCI(1,:), 2)
        if ~isempty(MEGMCI{i,t})
            % for j = 1:size(MEGMCI{i,t}, 1)
            %     for k = j:size(MEGMCI{i,t}, 1)
            %         PLIMCI{i,t}(j, k) = PLI(MEGMCI{i,t}(j,:), MEGMCI{i,t}(k,:));
            %     end
            % end
            PLIMCI{i,t} = PLI(MEGMCI{i,t}, MEGMCI{i,t});
        end
        t
    end
    i
end
%
%
% for i = 1:size(MEGAD, 1)
%     for t = 1:size(MEGAD(1,:), 2)
%         PLIAD{i,t} = triu(PLIAD{i,t})+tril(transpose(PLIAD{i,t}));
%     end
% end
%
%
% for i = 1:size(MEGHC, 1)
%     for t = 1:size(MEGHC(1,:), 2)
%         PLIHC{i,t} = triu(PLIHC{i,t})+tril(transpose(PLIHC{i,t}));
%     end
% end
%
% for i = 1:size(MEGMCI, 1)
%     for t = 1:size(MEGMCI(1,:), 2)
%         PLIMCI{i,t} = triu(PLIMCI{i,t})+tril(transpose(PLIMCI{i,t}));
%     end
% end

%% EEG data load

Path = 'C:/Users/MIQDAD/Desktop/PhD/PROPOSED Method/EEG_AD/OpenNeuroDatasets/ds004504-main/ds004504-main/derivatives';
FileList = dir(fullfile(Path))
for iFile = 3:numel(FileList)
    File = fullfile(FileList(iFile).folder, FileList(iFile).name);
    FileList2 = dir(fullfile(File));
    for k = 3:numel(FileList2)-1
        File = fullfile(FileList2(k+1).folder, FileList2(k+1).name);
        EEGFile{iFile-2, k-2} = load('-mat', File);
        EEG{iFile-2, k-2} = EEGFile{iFile-2, k-2}.data;
    end
end

EEGAD = EEG(1:36);
EEGHC = EEG(37:65);
EEGFD = EEG(66:88);
% 
% 
% %% PLI Calculation for EEG
PLIAD_eeg = {};
for i = 1:size(EEGAD, 1)
    for t = 1:floor(size(EEGAD{i}, 2)/Win)   %for each 10 sec segment
        % for j = 1:size(EEGAD{i}, 1)
        %     for k = j:size(EEGAD{i}, 1)
        %         PLIAD_eeg{i,t}(j, k) = PLI(EEGAD{i}(j, 1*t:Win*t), EEGAD{i}(k, 1*t:Win*t));
        %     end
        % end
        PLIAD_eeg{i,t} = PLI(EEGAD{i}(:, 1*t:Win*t), EEGAD{i}(:, 1*t:Win*t));
    end
    i
end
% 
% 
PLIHC_eeg = {};
for i = 1:size(EEGHC, 1)
    for t = 1:floor(size(EEGHC{i}, 2)/Win)   %for each 10 sec segment
        % for j = 1:size(EEGHC{i}, 1)
        %     for k = j:size(EEGHC{i}, 1)
        %         PLIHC_eeg{i,t}(j, k) = PLI(EEGHC{i}(j, 1*t:Win*t), EEGHC{i}(k, 1*t:Win*t));
        %     end
        % end
        PLIHC_eeg{i,t} = PLI(EEGHC{i}(:, 1*t:Win*t), EEGHC{i}(:, 1*t:Win*t));
    end
    i
end
% 
% 
PLIFD_eeg = {};
for i = 1:size(EEGFD, 1)
    for t = 1:floor(size(EEGFD{i}, 2)/Win)   %for each 10 sec segment
        % for j = 1:size(EEGFD{i}, 1)
        %     for k = j:size(EEGFD{i}, 1)
        %         PLIFD_eeg{i,t}(j, k) = PLI(EEGFD{i}(j, 1*t:Win*t), EEGFD{i}(k, 1*t:Win*t));
        %     end
        % end
        PLIFD_eeg{i,t} = PLI(EEGFD{i}(:, 1*t:Win*t), EEGFD{i}(:, 1*t:Win*t));
    end
    i
end
% 
% 
% for i = 1:size(EEGAD, 1)
%     for t = 1:size(EEGAD(1,:), 2)
%         PLIAD_eeg{i,t} = triu(PLIAD_eeg{i,t})+tril(transpose(PLIAD_eeg{i,t}));
%     end
% end
% 
% 
% for i = 1:size(EEGHC, 1)
%     for t = 1:size(EEGHC(1,:), 2)
%         PLIHC_eeg{i,t} = triu(PLIHC_eeg{i,t})+tril(transpose(PLIHC_eeg{i,t}));
%     end
% end
% 
% for i = 1:size(EEGFD, 1)
%     for t = 1:size(EEGFD(1,:), 2)
%         PLIFD_eeg{i,t} = triu(PLIFD_eeg{i,t})+tril(transpose(PLIFD_eeg{i,t}));
%     end
% end
% 
% 
% 
for i = 1:size(EEG_test, 1)
    for t = 1:floor(size(EEG_test, 2)/Win)   %for each 10 sec segment
        for j = 1:size(EEG_test, 1)
            for k = j:size(EEG_test, 1)
                PLIHC_eeg_test{t}(j, k) = PLI(EEG_test(j, 1*t:Win*t), EEG_test(k, 1*t:Win*t));
            end
        end
    end
    i
end
% 
% 

PLIHC_eeg_test = PLI(EEG_test, EEG_test);



for t = 1:size(PLIHC_eeg_test, 2)
    PLIHC_eeg_test{t} = triu(PLIHC_eeg_test{t})+tril(transpose(PLIHC_eeg_test{t}));
end

PLIHC_eeg_test = triu(PLIHC_eeg_test)+tril(transpose(PLIHC_eeg_test));

