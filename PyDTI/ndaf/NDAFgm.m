function [Zu,Zi,rnllk]=NDAFgm(A,Lu,Li,C,lambdau,lambdai,Z0u,Z0i,eta,maxitr,showevery)
% [Zu,Zi,rnllk]=NDAFgm(A,Lu,Li,C,lambdau,lambdai,Z0u,Z0i,maxitr)
% 
% Non-negative Diffusive Affinity Factorization, The General Model
% 
% A: adjacency matrix of User-Item network (MxN sparse/full, binary)
% Lu: a Laplacian matrix of user network (MxM sparse/full, real-value)
% Li: a Laplacian matrix of item network (NxN sparse/full, real-value)
% 
% Zu: user affinity matrix (MxK non-negative matrix)
% Zi: item affinity matrix (NxK non-negative matrix),
% 
% rnllk: the regularized negative log-likelihood of the NDAF
%
% M: number of users
% N: number of item
% K: number of latent factors (dimension of affinity vectors)
% 
% C: upper bound for ||Zu|| and ||Zi||
% 
% lambdau: regularization coefficient for Zu smoothing
% lambdai: regularization coefficient for Zi smoothing
%

%% Conventions and Initializations
include_softplus

[M,N]=size(A);
[U,I]=find(A);
Zu=Z0u; Zi=Z0i;
itr=0;

while 1
    sZu = sum(Zu); sZi = sum(Zi);
    B = dot(Zu(U,:),Zi(I,:),2);
    lG = sum(isoftplus(B)) - dot(sZu,sZi,2);
    LZu = Lu*Zu;
    hU = sum(dot(Zu,LZu)); % trace(Zu'*Lu*Zu);
    LZi = Li*Zi;
    hI = sum(dot(Zi,LZi)); % trace(Zi'*Li*Zi);
    rnllk = lambdau/2 * hU + lambdai/2 * hI - lG;
    B = sparse(U,I,1./(1-exp(-B)),M,N);
    gZu = bsxfun(@minus,(lambdau * LZu) , (B  * bsxfun(@minus,Zi , sZi)));
    gZi = bsxfun(@minus,(lambdai * LZi) , (B' * bsxfun(@minus,Zu , sZu)));
    
    %% A gradient descent update
    Z0u=Zu;
    Z0i=Zi;
    Zu = Zu - eta * gZu;
    Zi = Zi - eta * gZi;
    %% Simplex projection; project on {Z | Z>=0 && sum_k Zu<=C for each u && sum_k Zi<=C for each i}
    Zu = simplexproj(Zu,C); Zu(:,end)=max(0.01,Zu(:,end));
    Zi = simplexproj(Zi,C); Zi(:,end)=max(0.01,Zi(:,end));
    
    itr=itr+1;
    if mod(itr,showevery)==0, fprintf('itr = %d \t rnllk = %f\n',itr,rnllk); end
    %% Stopping criteria
    if itr==maxitr, break, end
    if max(abs(Zu(:)-Z0u(:)))<1e-6 && max(abs(Zi(:)-Z0i(:)))<1e-6, break, end
end
