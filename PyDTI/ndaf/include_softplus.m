% A numerically stable implementation for the softplus function and its inverse.
% By definition, softplus(x)=log(exp(x)+1), and isoftplus(x)=log(exp(x)-1).
softplus = @(x) (max(0,x)+log(1+exp(-abs(x))));
isoftplus= @(x) (x+log(max(0,1-exp(-x))));
