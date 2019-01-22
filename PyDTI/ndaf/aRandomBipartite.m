function [A,Zu,Zi]=aRandomBipartite(m,n,k,d)
if nargin<3 || isempty(k)
    k=10;
end
if nargin<4 || isempty(d)
    d=20;
end
w=0.7*randn(k,d); w(abs(w)<1)=0;
f=rand(d);
Zu=w(randi(k,m,1),:)*f+randn(m,d)/4;
f=f+randn(d)/4;
Zi=w(randi(k,n,1),:)*f+randn(n,d)/4;
Zu=Zu./((1+0.2*randn(size(Zu,1),1)).*sqrt(dot(Zu,Zu,2)));
Zi=Zi./((1+0.2*randn(size(Zi,1),1)).*sqrt(dot(Zi,Zi,2)));
A=sparse(Zu*Zi'>1);

% 
% [A,zu,zi]=aRandomBipartite(1000,200);
% Su=(1+corr(zu'))/2; Si=(1+corr(zi'))/2;
% [Zu,Zi,rnllk]=NDAF('lap',A,Su,Si,[],[],0,[],[],100);
% [recall,precision,f1,auroc,aupr,trshld]=lperfmea(A,Zu*Zi')
