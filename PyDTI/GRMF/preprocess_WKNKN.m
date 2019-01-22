function Y=preprocess_WKNKN(Y,Sd,St,K,eta)
%preprocess_WKNKN preprocesses the interaction matrix Y by replacing each
%of the 0's (i.e. presumed non-interactions) with a continuous value
%between 0 and 1. For each 0, the K nearest known drugs are used to infer
%a value, the K nearest known targets are used to infer another value, and
%then the average of the two values is used to replace that 0.
%
% Y = preprocess_WKNKN(Y,Sd,St,K,eta)
%
% INPUT:
%  Y:   matrix to be modified
%  Sd:  pairwise row similarities matrix
%  St:  pairwise column similarities matrix
%  K:   number of nearest known neighbors to use
%  eta: decay rate
%
% OUTPUT:
%  Y:   the modified matrix

    % decay values to be used in weighting similarities later
    eta = eta .^ (0:K-1);

    y2_new1 = zeros(size(Y));
    y2_new2 = zeros(size(Y));

    empty_rows = find(any(Y,2) == 0);   % get indices of empty rows
    empty_cols = find(any(Y)   == 0);   % get indices of empty columns

    % for each drug i...
    for i=1:length(Sd)
        drug_sim = Sd(i,:); % get similarities of drug i to other drugs
        drug_sim(i) = 0;    % set self-similiraty to ZERO

        indices  = 1:length(Sd);    % ignore similarities 
        drug_sim(empty_rows) = [];  % to drugs of 
        indices(empty_rows) = [];   % empty rows

        [~,indx] = sort(drug_sim,'descend');    % sort descendingly
        indx = indx(1:K);       % keep only similarities of K nearest neighbors
        indx = indices(indx);   % and their indices

        % computed profile of drug i by using its similarities to its K
        % nearest neighbors weighted by the decay values from eta
        drug_sim = Sd(i,:);
        y2_new1(i,:) = (eta .* drug_sim(indx)) * Y(indx,:) ./ sum(drug_sim(indx));
    end

    % for each target j...
    for j=1:length(St)
        target_sim = St(j,:); % get similarities of target j to other targets
        target_sim(j) = 0;    % set self-similiraty to ZERO

        indices  = 1:length(St);        % ignore similarities 
        target_sim(empty_cols) = [];    % to targets of
        indices(empty_cols) = [];       % empty columns

        [~,indx] = sort(target_sim,'descend');  % sort descendingly
        indx = indx(1:K);       % keep only similarities of K nearest neighbors
        indx = indices(indx);   % and their indices

        % computed profile of target j by using its similarities to its K
        % nearest neighbors weighted by the decay values from eta
        target_sim = St(j,:);
        y2_new2(:,j) = Y(:,indx) * (eta .* target_sim(indx))' ./ sum(target_sim(indx));
    end

    % average computed values of the modified 0's from the drug and target
    % sides while preserving the 1's that were already in Y 
    Y = max(Y,(y2_new1 + y2_new2)/2);

end