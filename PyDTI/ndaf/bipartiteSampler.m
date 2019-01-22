function [U,I]=bipartiteSampler(A,gltUsize,gltIsize,T,U,I)
if nargin<2 || isempty(gltUsize)
    gltUsize=16;
end
if nargin<3 || isempty(gltIsize)
    gltIsize=4;
end
if nargin<4 || isempty(T)
    T=10;
end
if nargin<6 || isempty(U) || isempty(I)
    [U,I]=find(A,randi(nnz(A)));
    U=U(end);
    I=I(end);
end

Usize=length(U);
Isize=length(I);
E=2*sum(sum(A(U,I)))-Usize*Isize;
A(U,I)=0;
NU=any(A(U,:),1);
NI=any(A(:,I),2);
while T>0.2
    deadlock=true;
    if Usize<gltUsize
        u=find(NI);
        if ~isempty(u)
            deadlock=false;
            u=u(randi(numel(u)));
            Enew=E+2*sum(A(u,I))-Isize;
            d=abs(Enew)-abs(E);
            if d<=0 || rand<=exp(-d/T)
                E=Enew;
                A(u,I)=0;
                U=[U,u]; Usize=Usize+1;
                NU=or(NU,A(u,:));
            end
        end
    end
    if Isize<gltIsize
        i=find(NU);
        if ~isempty(i)
            deadlock=false;
            i=i(randi(numel(i)));
            Enew=E+2*sum(A(U,i))-Usize;
            d=abs(Enew)-abs(E);
            if d<=0 || rand<=exp(-d/T)
                E=Enew;
                A(U,i)=0;
                I=[I,i]; Isize=Isize+1;
                NI=or(NI,A(:,i));
            end
        end
    end
    if deadlock, break, end
    T=0.9*T;
end
