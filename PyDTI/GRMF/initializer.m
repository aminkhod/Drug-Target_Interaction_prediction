function [A,B]=initializer(Y,K)
%initializer initializes the A and B latent feature matrices for the
%CMF/GRMF/WGRMF algorithms.
%
% [A,B] = initializer(Y,K)
%
% INPUT:
%  Y:   interaction matrix to be decomposed into latent feature matrices
%  K:   number of latent features
%
% OUTPUT:
%  A:   latent feature matrix for drugs
%  B:   latent feature matrix for targets

    [u,s,v] = svds(Y,K);
    A = u*(s^0.5);
    B = v*(s^0.5);

end