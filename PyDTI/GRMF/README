---------------------------------------------------------------------------

*** How to run GRMF? ***

The starting point for running GRMF is:

RUN_Cross_Validation.m
    This is for estimating the prediction performance of GRMF using cross 
    validation. More precisely, 5 repetitions of 10-fold cross validation 
    are performed, and then the AUPRs from the five repetitions are 
    averaged to give the final AUPR.

To run, simply access the above file in MATLAB, and press F5.


In the above file, there are many options that may be set, including:

> classifier
> use_WKNKN
> cv_setting
> cross validation parameters: m, n

---------------------------------------------------------------------------

*** Algorithm entry points ***

The main entry scripts for the different algorithms are:

> alg_blm_nii.m
> alg_rls_wnn.m
> alg_cmf.m
> alg_grmf.m
> alg_wgrmf.m
> alg_wknkn.m

Some parameters may be manually set to a fixed value, while others are 
estimated via nested CV. These parameters may be found in the above files, 
or if their values are determined by nested CV, they may be found in 
scripts that have the suffix "parest_":

> alg_grmf_parest.m
> alg_cmf_parest.m
> alg_rls_wnn_parest.m

However, for WKNKN in particular, the parameters may be found in 
RUN_Cross_Validation.m as WKNKN may be used as a preprocessing method by 
any of the other methods.

---------------------------------------------------------------------------

*** Output Description ***

In each data set, each repetition of 10-fold CV will have something similar
to the following printed on screen

==========================
n-fold experiment start:  	12 : 34 : 57.222    11-Dec-2015
****k100		2	0.1	0.1		0.671				TIME:    12 : 35 : 42.946    11-Dec-2015
****k100		2	0.1	0.1		0.746				TIME:    12 : 36 : 28.966    11-Dec-2015
****k100		2	0.1	0.1		0.87				TIME:    12 : 37 : 14.721    11-Dec-2015
****k100		2	0.1	0.1		0.818				TIME:    12 : 37 : 59.977    11-Dec-2015
****k100		2	0.1	0.1		0.824				TIME:    12 : 38 : 45.388    11-Dec-2015
****k100		2	0.1	0.1		0.767				TIME:    12 : 39 : 30.566    11-Dec-2015
****k100		2	0.1	0.1		0.864				TIME:    12 : 40 : 15.588    11-Dec-2015
****k100		2	0.1	0.1		0.797				TIME:    12 : 41 : 0.921    11-Dec-2015
****k100		2	0.1	0.1		0.81				TIME:    12 : 41 : 46.77    11-Dec-2015
****k100		2	0.1	0.1		0.625				TIME:    12 : 42 : 32.416    11-Dec-2015
n-fold experiment end:  	12 : 42 : 32.416    11-Dec-2015

      AUC: 0.930428
     AUPR: 0.778159
==========================

> The start and end times of the 10-fold CV are printed on screen.

> Each of the 10 folds takes its turn being the test set, resulting in 10 
  CV experiments. For each of these experiments, a line starting with 
  '****' holds the parameters estimated, and the last value to the right 
  (before the printed time) is the AUPR.

> After the 10-fold CV is over, the AUPRs from the 10 folds are averaged to
  give the average AUPR of the 10-fold CV. The average AUC is also given.


Finally, the AUPRs from the 5 repetitions of 10-fold CV are averaged to 
give the final AUPR of the data set.

---------------------------------------------------------------------------

*** Additional notes ***

How to set eta via cross validation in RLS-WNN??
> In the 'alg_rls_wnn.m' script, uncomment the following line:
    %eta = alg_rls_wnn_parest(Y,Sd,St,cv_setting,nr_fold,left_out);

In the CMF and GRMF files, be aware of the difference between the small 'k'
(rank of latent feature matrices A and B) and the capital 'K' (the number 
of known nearest neighbors for the WKNKN preprocessing method)

---------------------------------------------------------------------------

*** Relevant Publication ***

Drug-target interaction prediction with graph-regularized matrix factorization
Ali Ezzat, Peilin Zhao, Min Wu, Xiao-Li Li and Chee-Keong Kwoh

---------------------------------------------------------------------------