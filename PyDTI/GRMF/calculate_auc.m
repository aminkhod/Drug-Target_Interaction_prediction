function AUC=calculate_auc(targets,predicts)
%calculate_auc calculates the area under the Precission/recall curve
%
% AUC = calculate_auc(targets,predicts)
%
% INPUT:
%  targets:     actual labels
%  predicts:    prediction scores
%
% OUTPUT
%  AUC:        area under the ROC curve
%
% Borrowed from code of:
% Twan van Laarhoven, Sander B. Nabuurs, Elena Marchiori,
% (2011) Gaussian interaction profile kernels for predicting drug–target interaction
% http://cs.ru.nl/~tvanlaarhoven/drugtarget2011/

	if nargin > 1
		[~,i] = sort(predicts,'descend');
		targets = targets(i);
	end
	
	%for i=1:n
	%	if targets(i)
	%		goods = goods + 1
	%	else
	%		auc = auc + goods;
	%	end
	%end
	cumsums = cumsum(targets);
	AUC = sum(cumsums(~targets));
	pos = sum(targets);
	neg = sum(~targets);
	if pos == 0, warning('Calculate auc: no positive targets'); end
	if neg == 0, warning('Calculate auc: no negative targets'); end
	AUC = AUC / (pos * neg + eps);
end