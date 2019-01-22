function [recall,precision,f1,auroc,aupr,trshld]=lperfmea(A,B)
% A : ground truth (0/1 adjacency matrix of the bipartite graph), logical
% B : prediction

P=nnz(A); N=numel(A)-P;
[~,I]=sort(B(:));
x=cumsum(A(I));
y=(1:numel(A))'-x;
fpr=1-y/N;
tpr=1-x/P;
auroc=(A(I)'*y)/(N*P);
subplot(2,1,1); plot(fpr,tpr); ylim([0,1.05]); box off; grid on
xlabel('FPR'); ylabel('TPR'); title(['AUROC=' num2str(auroc)]);

pr=(P-x)./(numel(A)-0.5:-1:0.5)';
aupr=-trapz(tpr,pr);
subplot(2,1,2); plot(tpr,pr); ylim([0,1.05]); box off; grid on
xlabel('Recall'); ylabel('Precision'); title(['AUPR=' num2str(aupr)]);

[f1,itrshld]=max(2*pr.*tpr./(pr+tpr));
trshld=B(I(itrshld));
recall=tpr(itrshld);
precision=pr(itrshld);
