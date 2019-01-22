function A=graphSymmRWLap(A)
n=size(A,1);
A(1:n+1:end)=0;
A=bsxfun(@rdivide,A,bsxfun(@plus,sum(A,1),bsxfun(@rdivide,A,sum(A,2))));
A=diag(sum(A))-A;
