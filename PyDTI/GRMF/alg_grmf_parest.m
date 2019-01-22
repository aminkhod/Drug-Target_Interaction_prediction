function [k,lambda_l,lambda_d,lambda_t,p,num_iter]=alg_grmf_parest(Y,Sd,St,cv_setting,nr_fold,left_out,use_WKNKN,K,eta,use_W_matrix)
%alg_grmf_parest is a helper function of GRMF/WGRMF that estimates the
%parameters: K, lambda_l, lambda_d & lambda_t.

    %--------------------------------------------------------------------

    % ranges of parameter values to be tested to identify best combination
    range_k          = [  50   100   ];
    range_lambda_l   = [2^-2   2^-1   2^0   2^1];
    range_lambda_dt  = [   0  10^-4  10^-3  10^-2  10^-1];

    %nearest neighbor graph (for sparsification of similarity matrices)
    p = 5;

    % stopping criterion (number of iterations)
    num_iter = 2;

    %--------------------------------------------------------------------

    test_ind = get_test_indices(Y,cv_setting,left_out); % indices of the test set samples
    folds = get_folds(Y,cv_setting,nr_fold,left_out);   % folds of the CV done on the training set

    %--------------------------------------------------------------------

    y2s    = cell(1,nr_fold);
    Ws     = cell(1,nr_fold);
    initAs = cell(length(range_k),nr_fold);
    initBs = cell(length(range_k),nr_fold);
    for i=1:nr_fold
        y2 = Y;
        y2(folds{i}) = 0;  % folds{i} is the validation set
        if use_WKNKN
            y2 = preprocess_WKNKN(y2,Sd,St,K,eta);   % preprocessing Y
        end
        y2s{i} = y2;
        
        for k=1:length(range_k)
            [initAs{k,i},initBs{k,i}] = initializer(y2,range_k(k));
        end

        W = ones(size(y2));
        W(test_ind) = 0;
        W(folds{i}) = 0;
        Ws{i} = W;
    end

    % preprocessing Sd & St
    Sd = preprocess_PNN(Sd,p);
    St = preprocess_PNN(St,p);

    % Laplacian Matrices
    Dd = diag(sum(Sd));
    Dt = diag(sum(St));
    Ld = Dd - Sd;
    Ld = (Dd^(-0.5))*Ld*(Dd^(-0.5));
    Lt = Dt - St;
    Lt = (Dt^(-0.5))*Lt*(Dt^(-0.5));

    %--------------------------------------------------------------------

    best_AUPR = -Inf;
    for k=1:length(range_k)
        for lambda_l=range_lambda_l
            for lambda_d=range_lambda_dt
                for lambda_t=range_lambda_dt

                    % get overall AUPR for current parameter combination
                    AUPRs = zeros(nr_fold,1);
                    for i=1:nr_fold
                        y2 = y2s{i};        %
                        W = Ws{i};          % INITIALIZE
                        A = initAs{k,i};    %
                        B = initBs{k,i};    %


                        if ~use_W_matrix    % grmf                                                          %
                            [A,B] = alg_grmf_predict(y2,A,B,Ld,Lt,lambda_l,lambda_d,lambda_t,num_iter);     %
                        else                % wgrmf                                                         % PREDICT
                            [A,B] = alg_grmf_predict(y2,A,B,Ld,Lt,lambda_l,lambda_d,lambda_t,num_iter,W);   %
                        end                                                                                 %
                        y3 = A*B';                                                                          %


                        [~,AUPRs(i)] = returnEvaluationMetrics(Y(folds{i}),y3(folds{i}));   % EVALUATE
                    end
                    aupr_res = mean(AUPRs);


                    % keep parameter combination if it beats current best
                    if best_AUPR < aupr_res
                        best_AUPR = aupr_res;

                        best_k = range_k(k);
                        best_lambda_l = lambda_l;
                        best_lambda_d = lambda_d;
                        best_lambda_t = lambda_t;
                    end
                    
                end
            end
        end
    end

    %--------------------------------------------------------------------

    % return best parameters

           k = best_k;
    lambda_l = best_lambda_l;
    lambda_d = best_lambda_d;
    lambda_t = best_lambda_t;

    %--------------------------------------------------------------------

end