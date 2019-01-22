function [AUC,AUPR]=returnEvaluationMetrics(Y,scores)
%returnEvaluationMetrics computes the AUC and AUPR from a set of prediction
%scores and their corresponding labels.
%
% [AUC,AUPR] = returnEvaluationMetrics(Y,scores)
%
% INPUT:
%  Y:       labels
%  scores:  prediction scores
%
% OUTPUT:
%  AUC:     area under ROC curve
%  AUPR:    area under precision-recall curve

    AUC = calculate_auc(Y(:),scores(:));
    AUPR = calculate_aupr(Y(:),scores(:));

end