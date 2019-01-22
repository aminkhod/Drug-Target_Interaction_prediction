function folds=get_folds(Y,cv_setting,nr_fold,left_out)
%get_folds is a helper function that gets the indices of the training
%samples that belong to the different 'nr_fold' folds.
%
% folds = get_folds(Y,cv_setting,nr_fold,left_out)
%
% INPUT:
%  Y:           interaction matrix
%  cv_setting:  cross validation setting ('cv_d', 'cv_t' or 'cv_p')
%  nr_fold:     number of folds in cross validation experiment
%  left_out:    if cv_setting=='cv_d' --> left_out is 'drug' indices that are left out
%               if cv_setting=='cv_t' --> left_out is 'target' indices that are left out
%               if cv_setting=='cv_p' --> left_out is 'drug-target pair' indices that are left out
%
% OUTPUT:
%  folds:    the folds indices

    % get training set
    [num_drugs,num_targets] = size(Y);
    if strcmp(cv_setting,'cv_p')        % 'left_out' is left-out pairs
        trainingSet = 1:numel(Y);
        trainingSet(left_out) = [];
    elseif strcmp(cv_setting,'cv_d')    % 'left_out' is left-out drugs
        trainingSet = 1:size(Y,1);
        trainingSet(left_out) = [];
    elseif strcmp(cv_setting,'cv_t')    % 'left_out' is left-out targets
        trainingSet = 1:size(Y,2);
        trainingSet(left_out) = [];
    end
    len = length(trainingSet);
    rand_ind = randperm(len);
    rand_ind = trainingSet(rand_ind);


    % get the folds for 10-fold cross validation on the training set to
    % estimate parameters
    folds = cell(nr_fold);
    for i=1:nr_fold
        if strcmp(cv_setting,'cv_p')
            folds{i} = rand_ind((floor((i-1)*len/nr_fold)+1:floor(i*len/nr_fold))');

        elseif strcmp(cv_setting,'cv_d')
            folds_i_drugs = rand_ind((floor((i-1)*len/nr_fold)+1:floor(i*len/nr_fold))');
            folds{i} = zeros(length(folds_i_drugs),num_targets);
            for j=1:length(folds_i_drugs)
                curr_left_out_drug = folds_i_drugs(j);
                folds{i}(j,:) = ((0:(num_targets-1)) .* num_drugs) + curr_left_out_drug;
            end
            folds{i} = reshape(folds{i},numel(folds{i}),1);

        elseif strcmp(cv_setting,'cv_t')
            folds_i_targets = rand_ind((floor((i-1)*len/nr_fold)+1:floor(i*len/nr_fold))');
            folds{i} = zeros(num_drugs,length(folds_i_targets));
            for j=1:length(folds_i_targets)
                curr_left_out_target = folds_i_targets(j);
                folds{i}(:,j) = (1:num_drugs)' + ((curr_left_out_target-1)*num_drugs);
            end
            folds{i} = reshape(folds{i},numel(folds{i}),1);
        end
    end
    
end