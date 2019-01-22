function A=graphSymmNormaLap(A)
n=size(A,1);
A(1:n+1:end)=0;
D=1./sqrt(sum(A));
A=speye(n)-D'.*A.*D;
