function A=graphLaplacian(A)
n=size(A,1);
A(1:n+1:end)=0;
A=diag(sum(A))-A;
