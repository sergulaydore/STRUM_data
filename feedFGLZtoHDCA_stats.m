clc; clear all; %close all;
%%
for sub = 1:5
    clearvars -except sub
    expNo = ['814';'741';'812';'766';'763'];
    participant = [1 0 1 0 1];
    expNo = expNo(sub,:);
    participant =participant(sub);
    if participant ==1
        participant_name = 'CESlab102';
    else
        participant_name = 'install';
    end
    %% Load EEG Data
    tstart = [-1:0.1:1.8];
    tend = [-0.8:0.1:2];
    stimcue = 1;
    saveset = 0;
    
    folderName = sprintf('/home/turing/Documents/140501_SCCN_Data/%s/EEG_Data/',expNo);
    tempc = dir([folderName sprintf('*%s_Correct_64_Chan_EEGOnly_filt_1to30_Cleaned.set',participant_name)]);
    EEG_correct = pop_loadset('filename',{tempc.name},'filepath',folderName);
    numChan = size(EEG_correct.data,1);
    for stimcue = 1:2
        for i = 1:length(tstart)
            %close all
            if stimcue == 2
                folder_fastGLZ_results = sprintf(['/home/vnc/Documents/UCSD/SCCN_Scripts/FastGLZ'...
                    '/FGLZ_cue_only_more_overlap/Experiment_%s_Participant_%i_betterData_cueonly/timewin_%f_%f_eegFaSTGLZ/']...
                    ,expNo,participant,tstart(i),tend(i));
            else
                folder_fastGLZ_results = sprintf(['/home/vnc/Documents/UCSD/SCCN_Scripts/FastGLZ'...
                    '/FGLZ_stim_only_more_overlap/Experiment_%s_Participant_%i_betterData_stimonly/timewin_%f_%f_eegFaSTGLZ/']...
                    ,expNo,participant,tstart(i),tend(i));
            end
            
            
            
            load([folder_fastGLZ_results 'params.mat']);
            
            idxPerm = 0; % this will be many but let's assume we use only one permutation
            filename_perm = sprintf(['final_results_permNum%.0f.mat'], idxPerm);
            load([folder_fastGLZ_results 'final_results/' filename_perm]); % loads permutation results
            
            distanceMat = sqrt( ( ones(size(meanprobsels)) - meanprobsels).^2 ...
                + ( ones(size(meanprobsels)) - predictionAccuracies).^2 );
            [x,y] = find(distanceMat == min( distanceMat(:)),1);
            %Az(i) = predictionAccuracies(x,y);
            filename_opt = sprintf(['results_alphaInd_%.0f_lambdaInd_%.0f'],x(1),y(1)); % loads optimization results
            load([folder_fastGLZ_results 'optimization_results/' filename_opt]);
            
            if stimcue == 2
                etas_fglz_cue(i,:) = etas_test{x,y};
                truth_fglz_cue = glzOpts.ps.y;
            else
                etas_fglz_stim(i,:) = etas_test{x,y};
                truth_fglz_stim = glzOpts.ps.y;
            end
            
        end
    end
    num_etas_fglzcue = size(etas_fglz_cue,1);
    num_etas_fglzstim = size(etas_fglz_stim,1);
    %% Load Pupil Data
    [epoch_pupil_correct,~,epoch_pupil_skipped,~,response_time,label_type,srate_pupil]=extract_pupil_data(expNo,participant);
    pupilwin = -1:6;
    for winind = 1:numel(pupilwin)-1
        etas_pupil_correct(winind,:) = squeeze(mean(epoch_pupil_correct(1,round((pupilwin(winind)+1)*srate_pupil:(pupilwin(winind+1)+1)*srate_pupil)+1,:),2))';
        etas_pupil_skipped(winind,:) = squeeze(mean(epoch_pupil_skipped(1,round((pupilwin(winind)+1)*srate_pupil:(pupilwin(winind+1)+1)*srate_pupil)+1,:),2))';
    end
    etas_pupil_correct = bsxfun(@minus,etas_pupil_correct,squeeze(mean(epoch_pupil_correct(1,15:30,:),2))');
    etas_pupil_skipped = bsxfun(@minus,etas_pupil_skipped,squeeze(mean(epoch_pupil_skipped(1,15:30,:),2))');
    etas_pupil = cat(2,etas_pupil_correct,etas_pupil_skipped);
    truth_pupil = [ones(size(etas_pupil_correct,2),1); zeros(size(etas_pupil_skipped,2),1)];
    num_etas_pupil = size(etas_pupil,1);
    
    %% Load RT Data
    etas_RT_correct = response_time(label_type==1);
    etas_RT_skipped = response_time(label_type==3);
    etas_RT = cat(2,etas_RT_correct,etas_RT_skipped);
    num_etas_RT = size(etas_RT,1);
    
    
    %% Load HR Data
    load(sprintf('Experiment_%s_Participant_%i_HeartRate.mat',expNo,participant));
    etas_HR_correct = HeartRate(label_type==1);
    etas_HR_skipped = HeartRate(label_type==3);
    etas_HR = cat(2,etas_HR_correct,etas_HR_skipped);
    num_etas_HR = size(etas_HR,1);
    
    %% Run Fastglz
    etas_pupil_zscored = zscore(etas_pupil,[],2);
    etas_fglz_stim_zscored = zscore(etas_fglz_stim,[],2);
    etas_fglz_cue_zscored = zscore(etas_fglz_cue,[],2);
    etas_HR_zscored = zscore(etas_HR,[],2);
    etas_RT_zscored = zscore(etas_RT,[],2);
    
    
    etas_HR_zscored = zscore(etas_HR,[],2);
    etas_RT_zscored = zscore(etas_RT,[],2);
    etas_pupil_zscored = zscore(etas_pupil,[],2);
    
    y = truth_pupil;
    multimodal = {'etas_cue','etas_cue_stim','etas_cue_stim_pupil','etas_cue_stim_pupil_HR','etas_stim','etas_pupil'};
    for moda = 1:numel(multimodal)
        switch multimodal{moda}
            case 'etas_cue'
                X = cat(1,etas_fglz_cue_zscored);
            case 'etas_cue_stim'
                X = cat(1,etas_fglz_cue_zscored,etas_fglz_stim_zscored);
            case 'etas_cue_stim_pupil'
                X = cat(1,etas_fglz_cue_zscored,etas_fglz_stim_zscored,etas_pupil_zscored);
            case 'etas_cue_stim_pupil_HR'
                X = cat(1,etas_fglz_cue_zscored,etas_fglz_stim_zscored,etas_pupil_zscored,etas_HR_zscored);
            case 'etas_stim'
                X = cat(1,etas_fglz_stim_zscored);
            case 'etas_pupil'
                X = cat(1,etas_pupil_zscored);
        end
        
        [p,n] = size(X);
        fprintf(1,'Size of data matrix:  p=%d (number of features), n=%d (number of trials)\n',p,n);
        nnzmax = size(X,1);
        % Setup the problems structure -- 100 permutations, each running 10-fold
        % cross-validation
        % This amounts to K=1010 problems (10-fold cross-validation on 100 permutations + non-permuted data)
        ps = probs(p,y,...
            'numCVs',1,...
            'numFolds',5,...
            'numBootstraps',500,...
            'numPermutations',0,...
            'balanceClasses',0);
        fprintf(1,'The total number of problems to be run=%d',ps.K);
        
        glzOpts = fastglz_optsset('linkFcn','binomial',...
            'alphas',[0],...
            'lambdas',logspace(3,-1,200),...
            'ps',ps,...
            'addBiasTerm',1,...
            'nnzMax',nnzmax ,...
            'ALMMU_scaleFact',500,...
            'saveOutDir',sprintf('./FGLZ_Multimodal_500Boots_5fold/Experiment_%s_Participant_%i/%s/',expNo,participant,multimodal{moda}));
        
        
        for alphaInd=1:numel(glzOpts.alphas)
            fastglz(X,glzOpts,[],alphaInd);
        end
        adjust_FastGLZ_results_dir(glzOpts.saveOutDir,numel(glzOpts.alphas),numel(glzOpts.lambdas));
        fastglz_results(glzOpts.saveOutDir);
    end
    
end

