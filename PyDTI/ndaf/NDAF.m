function [Zu,Zi,rnllk,uSmthns,iSmthns,Su,Si]=NDAF(Ltype,A,Su,Si,K,C,lambda,eta,maxitr,showevery)
% [Zu,Zi,rnllk]=NDAF(Ltype,A,Su,Si,K,C,lambda,eta,maxitr,showevery)
% 
% Non-negative Diffusive Affinity Factorization
% 
% Ltype: type of the Laplacian matrices used for model regularization;
%        'lap' : the graph Laplacian, D-S
%        'lsyn': symmetric normalized Laplacian, I-D^(-1/2)*S*D^(-1/2)
%        'lrwn': random walk normalized Laplacian, I-D^(-1)*S
%        'lsyr': symmetric random walk Laplacian, lap(S") where S"=D^(-1)*S+S*D^(-1)
%        'user': No calculation for Laplacian matrices is invoked, instead,
%                Su and Si are used as Laplacian matrices.
%
% A: adjacency matrix of User-Item network (MxN sparse/full, binary)
% Su: user similarity matrix (MxM sparse/full, real-value)
% Si: item similarity matrix (NxN sparse/full, real-value)
% 
% M: number of users
% N: number of item
% K: number of latent factors (dimension of affinity vectors)
% 
% Zu: user affinity matrix (MxK non-negative matrix)
% Zi: item affinity matrix (NxK non-negative matrix)
% 
% rnllk: the regularized negative log-likelihood of the NDAF
%
% C: upper bound for ||Zu|| and ||Zi||
% 
% lambda: regularization coefficients,
%         [lambdau,lambdai], respectively, for Zu and Zi smoothing.
%
% eta: the learning rate
% 
% maxitr: maximum iteration
%
% showevery: show some info about optimization process, every `showevery` epochs.
%

Ltypes = {'lap','lsyn','lrwn','lsyr','user'};
Lfuncs = {@graphLaplacian, @graphSymmNormaLap, @graphRWNormaLap, @graphSymmRWLap, @(x) x};

%% Argument Validation and Default Settings
Ltype = find(strcmp(Ltype,Ltypes), 1);
if isempty(Ltype), fprintf(['Ltype must be one of ''%s''' repmat(', ''%s''',1,length(Ltypes)-1) '\n'],Ltypes{:}); return, end
Lfunc = Lfuncs{Ltype};

[M,N]=size(A);
A=logical(A);

[M1,M2]=size(Su);
if M1~=M || M2~=M, fprintf('Su must be a square matrix %d x %d\n',M,M); return, end
if any(any(Su~=Su')), fprintf('Su must be symmetric.\n'); return, end
Su=Lfunc(Su);

[N1,N2]=size(Si);
if N1~=N || N2~=N, fprintf('Si must be a square matrix %d x %d\n',N,N); return, end
if any(any(Si~=Si')), fprintf('Si must be symmetric.\n'); return, end
Si=Lfunc(Si);

if nargin<5 || isempty(K)
    K=30;
end
if nargin<6 || isempty(C)
    C=20;
end
if nargin<7 || isempty(lambda)
    lambda=1;
end
if isscalar(lambda)
    lambda=[lambda,lambda];
end
lambda=lambda(:);
if any(size(lambda)~=[2,1]), fprintf('lambda must be either a scalar or a two elements vector as [lambda_u, lambda_i]\n'); return, end

if nargin<8 || isempty(eta)
  eta=2e-4;
end
if nargin<9 || isempty(maxitr)
    maxitr=6000;
end
if nargin<10 || isempty(showevery)
    showevery=0;
end

include_softplus

[Zu,Zi,rnllk]=NDAFgm(A,Su,Si,C,lambda(1),lambda(2),abs(randn(M,K)/10),abs(randn(N,K)/10),eta,maxitr,showevery);

uSmthns = 1 / mean(mean(eps+abs(Su*Zu)./softplus(Zu)));
iSmthns = 1 / mean(mean(eps+abs(Si*Zi)./softplus(Zi)));
