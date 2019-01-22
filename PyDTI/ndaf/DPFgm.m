function [Zu,Zi,rllk]=DPFgm(A,Lu,Li,C,lambdau,lambdai,Z0u,Z0i,maxitr)
% [Zu,Zi,rllk]=DPFgm(A,Lu,Li,C,lambdau,lambdai,Z0u,Z0i,maxitr)
% 
% Diffusive Polarity Factorization, The General Model
% 
% A: adjacency matrix of User-Item network (MxN sparse/full, binary)
% Lu: a Laplacian matrix of user network (MxM sparse/full, real-value)
% Li: a Laplacian matrix of item network (NxN sparse/full, real-value)
% 
% Zu: user affinity matrix (MxK non-negative matrix)
% Zi: item affinity matrix (NxK non-negative matrix),
% 
% rllk: the regularized log-likelihood of the NDAF
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

eta = 0.001; % learning rate

itr=0;
Zu=Z0u;
Zi=Z0i;

while 1
    B = Zu*Zi';
    AZi = A*Zi;
    lG = sum(dot(Zu,AZi))-sum(sum(softplus(B))); % [U,I]=find(A); lG = sum(dot(Zu(U,:),Zi(I,:),2))-sum(sum(softplus(Zu*Zi')));
    LZu = Lu*Zu;
    hU = sum(dot(Zu,LZu)); % trace(Zu'*Lu*Zu);
    LZi = Li*Zi;
    hI = sum(dot(Zi,LZi)); % trace(Zi'*Li*Zi);
    rllk = lambdau/2 * hU + lambdai/2 * hI - lG;
    B = 1./(1+exp(-B)); % sigmoid(B)
    gZu = lambdau * LZu - (AZi - B * Zi);
    gZi = lambdai * LZi - (A - B)' * Zu;
    
    
    %% A gradient descent update
    Z0u=Zu;
    Z0i=Zi;
    Zu = Zu - eta * gZu;
    Zi = Zi - eta * gZi;
    %% Simplex projection; project on {Z | Z>=0 && sum_k Zu<=C for each u && sum_k Zi<=C for each i}
    Zu = simplexproj(Zu,C);
    Zi = simplexproj(Zi,C);
    
    %% Stopping criteria
    itr=itr+1;
    if itr==maxitr, break, end
    if max(abs(Zu(:)-Z0u(:)))<1e-6 && max(abs(Zi(:)-Z0i(:)))<1e-6, break, end
end
