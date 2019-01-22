 function predictR = runGRMF(args)
	Sd = args.Kx;
	St = args.Kz;
	Y = args.Y;
    cv = args.cv;
% --------------------------------------------------------------------------

%
% Author: 
% Ali Ezzat
% 


% Purpose:
% Perform drug-target interaction prediction in a cross-validation setting
% to estimate prediction performance of various prediction methods.
% (DEFAULT: 5 repetitions of 10-fold cross validation)
%

%clear   % clear workspace
%clc     % clear console screen

%diary off;  diary on;   % to save console output

%--------------------------------------------------------------------------

%*************************%
%* Adjustable Parameters *%
%*************************%

% The location of the folder that contains the data
%  path='..\data\datasets\';
%  ds=4;
% % the different datasets
%  datasets={'e','ic','gpcr','nr'};

% CLASSIFIER -------------------------------
%classifier='blm_nii';
%classifier='rls_wnn';
%classifier='cmf';
classifier='grmf';
%classifier='wgrmf';
%classifier='wknkn';
% ------------------------------------------

% Parameters and Options -------------------
%WKNKN
use_WKNKN = 0;      % 1=yes, 0=no
K = 5;              % number of K nearest known neighbors
eta = 0.7;          % decay rate (also used by WNN in RLS-WNN)

%weight matrix W (used by WGRMF and CMF)
% if strcmp(classifier,'wgrmf') || strcmp(classifier,'cmf')
if strcmp(classifier,'wgrmf')
    use_W_matrix = 1;
else
    use_W_matrix = 0;
end
% ------------------------------------------

% CROSS VALIDATION SETTING -----------------
% cv_setting = 'cv_d';    % DRUG PREDICTION CASE
% cv_setting = 'cv_t';    % TARGET PREDICTION CASE
% cv_setting = 'cv_p';    % PAIR PREDICTION CASE
% ------------------------------------------

% CROSS VALIDATION PARAMETERS --------------
m = 1;  % number of n-fold experiments (repetitions)
n = 10; % the 'n' in "n-fold experiment"
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

switch cv
    case 1, cv_setting= 'cv_p';
    case 2, cv_setting= 'cv_d';
    case 3, cv_setting= 'cv_t';
	case 4, cv_setting= 'cv_p';
end

% if use_WKNKN
%     fprintf('\nusing WKNKN: K=%i, eta=%g\n',K,eta);
% end
% fprintf('\n');

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

%for ds=[4 3 2 1]
    %disp('--------------------------------------------------------------');

%     fprintf('\nData Set: %s\n', datasets{ds});
% 
%     % LOAD DATA
%     [Y,Sd,St,Did,Tid]=getdata(path,datasets{ds});

    % PREDICT (+ print evaluation metrics)
%     predictR = crossValidation(Y',Sd,St,classifier,use_WKNKN,K,eta,use_W_matrix);

  
    predictR = crossValidation(Y,Sd,St,classifier,cv_setting,m,n,use_WKNKN,K,eta,use_W_matrix);
%     [x,y]=size(Y);
%     fprintf(x+string('') +y)
%     [x, y]=size(predictR);
%     fprintf(x+string('') +y)
   % disp('--------------------------------------------------------------');
 %   diary off;  diary on;
%end

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

%disp('==============================================================');
%diary off;
 end