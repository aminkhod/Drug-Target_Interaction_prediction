function test_ind=get_test_indices(Y,cv_setting,left_out)
%get_test_indices gets the test indices for the values in left_out
%
% test_ind = get_test_indices(Y,cv_setting,left_out)
%
% INPUT:
%  Y:           interaction matrix
%  cv_setting:  cross validation setting ('cv_d', 'cv_t' or 'cv_p')
%  left_out:    if cv_setting=='cv_d' --> left_out is 'drug' indices that are left out
%               if cv_setting=='cv_t' --> left_out is 'target' indices that are left out
%               if cv_setting=='cv_p' --> left_out is 'drug-target pair' indices that are left out
%
% OUTPUT:
%  test_ind:    the test indices corresponding to the values in left_out

    [num_drugs,num_targets] = size(Y);

    % 'left_out' is left-out interactions
    if strcmp(cv_setting,'cv_p')
        test_ind = left_out;

    % 'left_out' is left-out drugs
    elseif strcmp(cv_setting,'cv_d')
        test_ind = zeros(length(left_out),num_targets);
        for j=1:length(left_out)
            curr_left_out_drug = left_out(j);
            test_ind(j,:) = ((0:(num_targets-1)) .* num_drugs) + curr_left_out_drug;
        end
        test_ind = reshape(test_ind,numel(test_ind),1);

    % 'left_out' is left-out targets
    elseif strcmp(cv_setting,'cv_t')
        test_ind = zeros(num_drugs,length(left_out));
        for j=1:length(left_out)
            curr_left_out_target = left_out(j);
            test_ind(:,j) = (1:num_drugs)' + ((curr_left_out_target-1)*num_drugs);
        end
        test_ind = reshape(test_ind,numel(test_ind),1);
    end
    
end