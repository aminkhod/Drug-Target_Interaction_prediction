import getopt
import sys
import time

import cv_eval
from DTHybrid import DTHYBRID
from netcbp.netcbp import NetCBP
from blm import BLMNII
from brdti import BRDTI
from cmf import CMF
from daspfind import DASPFIND
from ddr import DDR
from dnilmf import DNILMF
from functions import *
from kbmf2k.kbmf import KBMF
from kronrlsmkl import KronRLsMKL
from netlaprls import NetLapRLS
from new_pairs import novel_prediction_analysis
from nrlmf import NRLMF
from vbmklmf import VBMKLMF
from wnngip import WNNGIP
from pudt.pudt import PUDT
from GRMF.GRMF import GRMF
from EnsambleDTI import EnsambleDTI
# from ndaf.NDAF import NDAF
def main(argv):
    try:
        opts, args = getopt.getopt(argv, "m:d:f:c:s:o:n:p", ["method=", "dataset=", "data-dir=", "cvs=", "specify-arg=", "method-options=", "predict-num=", "output-dir=", ])
    except getopt.GetoptError:
        sys.exit()

    data_dir = os.path.join(os.path.pardir, 'data')
    output_dir = os.path.join(os.path.pardir, 'output')
    cvs, sp_arg, model_settings, predict_num = 1, 1, [], 0
    seeds = [7771, 8367, 22, 1812, 4659]
    seedsOptPar = [156]

    # seeds = np.random.choice(10000, 5, replace=False)
    for opt, arg in opts:
        if opt == "--method":
            method = arg
        # if opt == "--dataset":
        #     dataset = arg
        if opt == "--data-dir":
            data_dir = arg
        if opt == "--output-dir":
            output_dir = arg
        if opt == "--cvs":
            cvs = int(arg)
        if opt == "--specify-arg":
            sp_arg = int(arg)
        if opt == "--method-options":
            model_settings = [s.split('=') for s in str(arg).split()]
        if opt == "--predict-num":
            predict_num = int(arg)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    # default parameters for each methods
    method = 'ensambledti'
    # sp_arg=0
    if method == 'nrlmf':
        args = {'c': 5, 'K1': 5, 'K2': 5, 'r': 50, 'lambda_d': 0.125, 'lambda_t': 0.125,
                'alpha': 0.25, 'beta': 0.125, 'theta': 0.5, 'max_iter': 100}
    if method == 'daspfind':
        args = {'alpha':2.26}
    if method == 'pudt':
        args = {""}
    # if method == 'ndaf':
    #     args = {""}
    if method == 'grmf':
        args = {""}
    if method == 'netcbp':
        args = {""}
    if (method == 'dnilmf'):
        args={""}
    if (method == 'dthybrid'):
        args = {""}
    if (method == 'kronrlsmkl'):
        args = {""}
    if (method == 'ddr'):
        args={""}
    if (method == 'vbmklmf'):
        args={""}

    if (method == 'brdti') | (method == 'inv_brdti') :
        args = {
            'D':100,
            'learning_rate':0.1,
            'max_iters' : 100,
            'simple_predict' :False,
            'bias_regularization':1,
            'global_regularization':10**(-2),
            "cbSim": "knn",
            'cb_alignment_regularization_user' :1,
            'cb_alignment_regularization_item' :1}

    if method == 'netlaprls':
        args = {'gamma_d': 10, 'gamma_t': 10, 'beta_d': 1e-5, 'beta_t': 1e-5}
    if method == 'blmnii':
        args = {'alpha': 0.7, 'gamma': 1.0, 'sigma': 1.0, 'avg': False}
    if method == 'wnngip':
        args = {'T': 0.8, 'sigma': 1.0, 'alpha': 0.8}
    if method == 'kbmf':
        args = {'R': 50}
    if method == 'ensambledti':
        args == {''}
    if method == 'cmf':
        args = {'K': 50, 'lambda_l': 0.5, 'lambda_d': 0.125, 'lambda_t': 0.125, 'max_iter': 30}

    for key, val in model_settings:
        args[key] = val

    #Multi threading
    import threading
	
    class myThread(threading.Thread):
        def __init__(self, thrID, method, dataset, data_dir, output_dir, cvs, sp_arg, model_settings, predict_num,
                     seeds, seedsOptPar,args):
            threading.Thread.__init__(self)
            self.thrID = thrID
            self.method = method
            self.dataset = dataset
            self.data_dir = data_dir
            self.output_dir = output_dir
            self.cvs = cvs
            self.sp_arg = sp_arg
            self.model_settings = model_settings
            self.predict_num = predict_num
            self.seeds = seeds
            self.seedsOptPar = seedsOptPar
            self.args = args

        def run(self):
            print("Starting thread " + str(self.thrID) + " with" + self.method + " on Dataset of " + self.dataset)
            # Get lock to synchronize threads
            #threadLock.acquire()
            thear(self.method, self.dataset, self.data_dir, self.output_dir, self.cvs, self.sp_arg,
                  self.model_settings, self.predict_num, self.seeds, self.seedsOptPar,args)
            # Free lock to release next thread
            #threadLock.release()
            print("Exiting thread " + str(self.thrID) + " with " + str(self.method) + " on Dataset of " + self.dataset)

    # threadLock = threading.Lock()
    threads = []

    # Create new threads


    thread1 = myThread(1, method, "e", data_dir, output_dir, cvs, sp_arg, model_settings, predict_num, seeds,
                       seedsOptPar, args)

    thread2 = myThread(2, method, "gpcr", data_dir, output_dir, cvs, sp_arg, model_settings, predict_num, seeds,
                        seedsOptPar, args)

    thread3 = myThread(3, method, "ic", data_dir, output_dir, cvs, sp_arg, model_settings, predict_num, seeds,
                        seedsOptPar, args)

    thread4 = myThread(4, method, "nr", data_dir, output_dir, cvs, sp_arg, model_settings, predict_num, seeds,
                        seedsOptPar, args)

    # thread1.start()
    # thread2.start()
    # thread3.start()
    thread4.start()


    # Add threads to thread list
    # threads.append(thread1)
    # threads.append(thread2)
    # threads.append(thread3)
    # threads.append(thread4)


#### Wait for all threads to complete
    for t in threads:
       t.join()
    print("Exiting Main Thread")

def thear(method, dataset, data_dir, output_dir, cvs, sp_arg, model_settings, predict_num, seeds, seedsOptPar,
          args):
    intMat, drugMat, targetMat = load_data_from_file( dataset, os.path.join(data_dir, 'datasets'))
    drug_names, target_names = get_drugs_targets_names(dataset, os.path.join(data_dir, 'datasets'))

    invert = 0
    if (method == 'inv_brdti')  :
        invert = 1
    if predict_num == 0:
        if cvs == 1:  # CV setting CVS1
            X, D, T, cv = intMat, drugMat, targetMat, 1
        if cvs == 2:  # CV setting CVS2
            X, D, T, cv = intMat, drugMat, targetMat, 0
        if cvs == 3:  # CV setting CVS3
            X, D, T, cv, invert = intMat.T, targetMat, drugMat, 0, 1
        if cvs == 4:
            X, D, T, cv = intMat,  drugMat, targetMat, 2
        cv_data = cross_validation(X, seeds, cv, invert, num = 10)

    if invert:
        X, D, T = intMat, drugMat, targetMat

        #cv_data_optimize_params = cross_validation(X, seedsOptPar, cv, invert, num=5)

    if sp_arg == 0 and predict_num == 0:
        if (method=="vbmklmf"):
            cv_eval.vbmklmf_cv_eval(method, dataset, cv_data, X, D, T, cvs, args)
        if (method == "ensambledti"):
            cv_eval.vbmklmf_cv_eval(method, dataset, cv_data, X, D, T, cvs, args)
        if method == 'netcbp':
            cv_eval.netcbp_cv_eval(method, dataset, cv_data, X, D, T, cvs, args)
        # if method == 'ndaf':
        #     cv_eval.ndaf_cv_eval(method, dataset, cv_data, X, D, T, cvs, args)
        if method == 'grmf':
            cv_eval.grmf_cv_eval(method, dataset, cv_data, X, D, T, cvs, args)
        if method == 'pudt':
            cv_eval.pudt_cv_eval(method, dataset, cv_data, X, D, T, cvs, args)
        if method == 'daspfind':
            cv_eval.daspfind_cv_eval(method, dataset, cv_data, X, D, T, cvs, args)
        if method == 'dnilmf':
            cv_eval.dnilmf_cv_eval(method, dataset, cv_data, X, D, T, cvs, args)
        if method == 'dthybrid':
            cv_eval.dthybrid_cv_eval(method, dataset, cv_data, X, D, T, cvs, args)
        if method == 'kronrlsmkl':
            cv_eval.kronrismkl_cv_eval(method, dataset, cv_data, X, D, T, cvs, args)
        if (method == 'brdti'):
            cv_eval.brdti_cv_eval(method, dataset, cv_data, X, D, T, cvs, args)
        if(method == 'ddr'):
            cv_eval.ddr_cv_eval(method, dataset, cv_data, X, D, T, cvs, args)
        if (method == 'brdti'):
            cv_eval.brdti_cv_eval(method, dataset, cv_data, X, D, T, cvs, args)
        if (method == 'inv_brdti'):
            cv_eval.brdti_cv_eval(method, dataset, cv_data, X, D, T, cvs, args)
        if method == 'nrlmf':
            cv_eval.nrlmf_cv_eval(method, dataset, cv_data, X, D, T, cvs, args)
        if method == 'netlaprls':
            cv_eval.netlaprls_cv_eval(method, dataset, cv_data, X, D, T, cvs, args)
        if method == 'blmnii':
            cv_eval.blmnii_cv_eval(method, dataset, cv_data, X, D, T, cvs, args)
        if method == 'wnngip':
            cv_eval.wnngip_cv_eval(method, dataset, cv_data, X, D, T, cvs, args)
        if method == 'kbmf':
            cv_eval.kbmf_cv_eval(method, dataset, cv_data, X, D, T, cvs, args)
        if method == 'cmf':
            cv_eval.cmf_cv_eval(method, dataset, cv_data, X, D, T, cvs, args)

    if sp_arg == 1 or predict_num > 0:
        tic = time.clock()
        if (method=="netcbp"):
            model = NetCBP()
        # if (method=="ndaf"):
        #     model = NDAF()
        if (method=="grmf"):
            model = GRMF(cv=cvs)
        if (method=="pudt"):
            model = PUDT(dataset=dataset)
        if (method=="vbmklmf"):
            model = VBMKLMF(dataset=dataset,cvs=cvs)
        if (method == 'dnilmf'):
            model = DNILMF(dataset=dataset)
        if (method == 'kronrlsmkl'):
            model = KronRLsMKL(dataset=dataset)
        if (method == 'dthybrid'):
            model = DTHYBRID(dataset=dataset)
        if (method == 'daspfind'):
            model = DASPFIND(alpha=args['alpha'])
        if (method == 'brdti')|(method == 'inv_brdti'):
            #model = BRDTI(D=args['D'],learning_rate= args['learning_rate'],max_iters=args['max_iters'],simple_predict=args['simple_predict'],bias_regularization=args['bias_regularization'],global_regularization=args['global_regularization'],cbSim=args['cbSim'],cb_alignment_regularization_user=args['cb_alignment_regularization_user'],cb_alignment_regularization_item=args['cb_alignment_regularization_item'])
            model = BRDTI(args)
        if method == 'nrlmf':
            model = NRLMF(cfix=args['c'], K1=args['K1'], K2=args['K2'], num_factors=args['r'], lambda_d=args['lambda_d'], lambda_t=args['lambda_t'], alpha=args['alpha'], beta=args['beta'], theta=args['theta'], max_iter=args['max_iter'])
        if method == 'ddr' :
            model = DDR(dataset=dataset,cv=cvs)
        if method == 'netlaprls':
            model = NetLapRLS(gamma_d=args['gamma_d'], gamma_t=args['gamma_t'], beta_d=args['beta_t'], beta_t=args['beta_t'])
        if method == 'blmnii':
            model = BLMNII(alpha=args['alpha'], gamma=args['gamma'], sigma=args['sigma'], avg=args['avg'])
        if method == 'wnngip':
            model = WNNGIP(T=args['T'], sigma=args['sigma'], alpha=args['alpha'])
        if method == 'kbmf':
           model = KBMF(num_factors=args['R'])
        if method == 'cmf':
            model = CMF(K=args['K'], lambda_l=args['lambda_l'], lambda_d=args['lambda_d'], lambda_t=args['lambda_t'], max_iter=args['max_iter'])
        if (method == 'ensambledti'):
            model = EnsambleDTI(args= args,dataset=dataset)
        cmd = str(model)
        if predict_num == 0:
            print("Dataset:"+dataset+" CVS:"+str(cvs)+"\n"+cmd)
            aupr_vec, auc_vec = train(model, cv_data, X, D, T,cvs,dataset)
            aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
            auc_avg, auc_conf = mean_confidence_interval(auc_vec)
            print("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock()-tic))
            write_metric_vector_to_file(auc_vec, os.path.join(output_dir, method+"_auc_cvs"+str(cvs)+"_"+dataset+".txt"))
            write_metric_vector_to_file(aupr_vec, os.path.join(output_dir, method+"_aupr_cvs"+str(cvs)+"_"+dataset+".txt"))
        elif predict_num > 0:
            print("Dataset:"+dataset+"\n"+cmd)
            seed = 7771 if method == 'cmf' else 22
            model.fix_model(intMat, intMat, drugMat, targetMat, seed)
            x, y = np.where(intMat == 0)
            scores = model.predict_scores(zip(x, y), 5)
            ii = np.argsort(scores)[::-1]
            predict_pairs = [(drug_names[x[i]], target_names[y[i]], scores[i]) for i in ii[:predict_num]]
            new_dti_file = os.path.join(output_dir, "_".join([method, dataset, "new_dti.txt"]))
            novel_prediction_analysis(predict_pairs, new_dti_file, os.path.join(data_dir, 'biodb'))

if __name__ == "__main__":

    main(sys.argv[1:])
#python PyDTI.py --method="cmf" --dataset="nr" --cvs=2

