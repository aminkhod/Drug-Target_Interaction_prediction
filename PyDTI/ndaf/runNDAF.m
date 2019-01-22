%  function predictR = runNDAF(args)
% 	Sd = args.Kx;
% 	St = args.Kz;
% 	Y2 = args.Y;

% --------------------------------------------------------------------------



%--------------------------------------------------------------------------

%*************************%
%* Adjustable Parameters *%
%*************************%



% CLASSIFIER -------------------------------
%classifier='blm_nii';
%classifier='rls_wnn';
%classifier='cmf';
% classifier='grmf';
%classifier='wgrmf';
%classifier='wknkn';
% ------------------------------------------



% CROSS VALIDATION SETTING -----------------
% cv_setting = 'cv_d';    % DRUG PREDICTION CASE
% cv_setting = 'cv_t';    % TARGET PREDICTION CASE
% cv_setting = 'cv_p';    % PAIR PREDICTION CASE
% ------------------------------------------

% CROSS VALIDATION PARAMETERS --------------
% m = 5;  % number of n-fold experiments (repetitions)
% n = 10; % the 'n' in "n-fold experiment"
% ------------------------------------------

%warning off     % to be used when many unnecessary warnings are being produced

%--------------------------------------------------------------------------

% Terminology:
% Y = Interaction matrix
% Sd = Drug similarity matrix
% St = Target similarity matrix


% disp('==============================================================');
% fprintf('\nClassifier Used: %s',classifier);
% switch cv_setting
%     case 'cv_d', fprintf('\nCV Setting Used: CV_d - New Drug\n');
%     case 'cv_t', fprintf('\nCV Setting Used: CV_t - New Target\n');
%     case 'cv_p', fprintf('\nCV Setting Used: CV_p - Pair Prediction\n');
% end


%     for ds=[4 3 2 1]
% 
%     The location of the folder that contains the data
    
    clear   % clear workspace
    clc     % clear console screen


     path='data\';
     ds=2;
    % the different datasets
    datasets={'e','ic','gpcr','nr'};
    fprintf('\nData Set: %s\n', datasets{ds});

    [Y,Sd,St,Did,Tid]=getdata(path,datasets{ds});
    load DTHybridW.txt   
    Y=Y';
    Y2=DTHybridW.*Y;
    K = 30;
    C = 10;
    lambda = [1,2];
    eta = 0.00008;
    maxitr = 8500;
    showevery = 0;
    Ltype = 'lsyr';
    [Zu,Zi,rnllk] = NDAF(Ltype,Y2,Sd,St,K,C,lambda,eta,maxitr,showevery);
    predictR=Zu*Zi';
    [recall,precision,f1,auroc,aupr,trshld]=lperfmea(Y,predictR);
    
    
%     end
%  end