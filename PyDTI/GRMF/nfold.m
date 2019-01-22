function [y3]=nfold(Y,Sd,St,pred_fn,nr_fold,seed,cv_setting,use_WKNKN,K,eta,use_W_matrix)
% function [y3]=nfold(Y,Sd,St,pred_fn,nr_fold,cv_setting,use_WKNKN,K,eta,use_W_matrix)
%nfold is a helper function of crossValidation.m. Depending on the
%specified CV setting (or scenario) and supplied "seed", it divides the
%interaction matrix into "nr_fold" folds, performs a cross validation
%experiment and then reports the results (AUPR/AUC) back to
%crossValidation.m.
%
% INPUT:
%  Y:           matrix to be modified
%  Sd:          pairwise row similarities matrix
%  St:          pairwise column similarities matrix
%  pred_fn:     function/script of algorithm to be used for DTI prediction
%  nr_fold:     number of folds in cross validation experiment
%  seed:        seed used for random sampling
%  cv_setting:  drug prediction case or target prediction case
%
% OUTPUT:
%  auc_res:     AUC result
%  aupr_res:    AUPR result
%

    [num_drugs,num_targets] = size(Y);
    if strcmp(cv_setting,'cv_p')
        len = numel(Y);
    elseif strcmp(cv_setting,'cv_d')
        len = num_drugs;
    elseif strcmp(cv_setting,'cv_t')
        len = num_targets;
    end
    rng('default')
    rng(seed);
    rand_ind = randperm(len);

    % TIME PRINT ----------------------------
%     fprintf('n-fold experiment start:  \t');    timeprint();
    % ---------------------------------------

%     AUCs  = zeros(1,nr_fold);
%     AUPRs = zeros(1,nr_fold);
    i=2;

%     for i=1:nr_fold
%         leave out random drug-target pairs
        if strcmp(cv_setting,'cv_p')
            test_ind = rand_ind((floor((i-1)*len/nr_fold)+1:floor(i*len/nr_fold))');
            left_out = test_ind;

        % leave out random entire drugs
        elseif strcmp(cv_setting,'cv_d')
            left_out_drugs = rand_ind((floor((i-1)*len/nr_fold)+1:floor(i*len/nr_fold))');
            test_ind = zeros(length(left_out_drugs),num_targets);
            for j=1:length(left_out_drugs)
                curr_left_out_drug = left_out_drugs(j);
                test_ind(j,:) = ((0:(num_targets-1)) .* num_drugs) + curr_left_out_drug;
            end
            test_ind = reshape(test_ind,numel(test_ind),1);
            left_out = left_out_drugs;

        % leave out random entire targets
        elseif strcmp(cv_setting,'cv_t')
            left_out_targets = rand_ind((floor((i-1)*len/nr_fold)+1:floor(i*len/nr_fold))');
            test_ind = zeros(num_drugs,length(left_out_targets));
            for j=1:length(left_out_targets)
                curr_left_out_target = left_out_targets(j);
                test_ind(:,j) = (1:num_drugs)' + ((curr_left_out_target-1)*num_drugs);
            end
            test_ind = reshape(test_ind,numel(test_ind),1);
            left_out = left_out_targets;
        end

        % predict with test set being left out
        y2 = Y;
%         y2(test_ind) = 0;   % test set = ZERO
%         fprintf('****');
       
        y3 = pred_fn(y2,Sd,St,cv_setting,nr_fold,left_out,use_WKNKN,K,eta,use_W_matrix); % predict!

%         % compute evaluation metrics based on obtained prediction scores
%         [AUCs(i), AUPRs(i)] = returnEvaluationMetrics(Y(test_ind)',y3(test_ind)');
%         fprintf('%.3g\t\t\t\tTIME:    ',AUPRs(i));  timeprint();
%         diary off;  diary on;
%     end

    % TIME PRINT ----------------------------
%     fprintf('n-fold experiment end:  \t');  timeprint();
%     % ---------------------------------------
% 
%     auc_res = mean(AUCs);
%     aupr_res = mean(AUPRs);
%     fprintf('\n');
%     fprintf('      AUC: %g\n',   auc_res);
%     fprintf('     AUPR: %g\n',   aupr_res);
%     disp('==========================');

end

% function timeprint()
%    clk = clock;
%    fprintf('%g : %g : %g    %s\n',clk(4:6),date);
% end