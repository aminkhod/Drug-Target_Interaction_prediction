%     for ds=[4 3 2 1]
% 
%     The location of the folder that contains the data
    clear   % clear workspace
    clc     % clear console screen


     path='data\';
     ds=4;
    % the different datasets
    datasets={'e','ic','gpcr','nr'};
    fprintf('\nData Set: %s\n', datasets{ds});

    [Y,Sd,St,Did,Tid]=getdata(path,datasets{ds});
    load W.txt   
    Y=Y';
    Y2=W.*Y;
    K = 30;
    C = 40;
    lambda = [0.06,0.1];
    eta = 0.00093;
    maxitr = 6000;
    showevery = 0;
    Ltype = 'lrwn';
    [Zu,Zi,rnllk] = NDAF(Ltype,Y2,Sd,St,K,C,lambda,eta,maxitr,showevery);
    predictR=Zu*Zi';
   [recall,precision,f1,auroc,aupr,trshld]=lperfmea(Y,predictR);
    
    
%     end