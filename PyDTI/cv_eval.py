import time

from DTHybrid import DTHYBRID
from netcbp.netcbp import NetCBP
from pudt.pudt import PUDT
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
from nrlmf import NRLMF
# from vbmklmf import VBMKLMF
from wnngip import WNNGIP
from GRMF import GRMF
# from ndaf.NDAF import NDAF
def vbmklmf_cv_eval(method, dataset, cv_data, X, D, T, cvs, para):
    max_auc, auc_opt = 0, []
    tic = time.clock()
    model = VBMKLMF(dataset,cvs)
    cmd = "Dataset:"+dataset+" CVS: "+str(cvs)+"\n"+str(model)
    print(cmd)
    aupr_vec, auc_vec = train(model, cv_data, X, D, T)
    aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
    auc_avg, auc_conf = mean_confidence_interval(auc_vec)
    print("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock()-tic))
    if auc_avg > max_auc:
        max_auc = auc_avg
        auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
    cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f\n" % (auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4])
    cmd +=" max auc:%.6f\n"%(max_auc)
    print(cmd)
def dthybrid_cv_eval(method, dataset, cv_data, X, D, T, cvs, para):
    max_auc, auc_opt = 0, []
    tic = time.clock()
    model = DTHYBRID(dataset)
    cmd = "Dataset:"+dataset+" CVS: "+str(cvs)+"\n"+str(model)
    print(cmd)
    aupr_vec, auc_vec = train(model, cv_data, X, D, T)
    aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
    auc_avg, auc_conf = mean_confidence_interval(auc_vec)
    print("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock()-tic))
    if auc_avg > max_auc:
        max_auc = auc_avg
        auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
    cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f\n" % (auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4])
    cmd +=" max auc:%.6f\n"%(max_auc)
    print(cmd)

def dnilmf_cv_eval(method, dataset, cv_data, X, D, T, cvs, para):
    max_auc, auc_opt = 0, []
    tic = time.clock()
    model = DNILMF(dataset)
    cmd = "Dataset:"+dataset+" CVS: "+str(cvs)+"\n"+str(model)
    print(cmd)
    aupr_vec, auc_vec = train(model, cv_data, X, D, T)
    aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
    auc_avg, auc_conf = mean_confidence_interval(auc_vec)
    print("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock()-tic))
    if auc_avg > max_auc:
        max_auc = auc_avg
        auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
    cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f\n" % (auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4])
    cmd +=" max auc:%.6f\n"%(max_auc)
    print(cmd)

def kronrismkl_cv_eval(method, dataset, cv_data, X, D, T, cvs, para):
    max_auc, auc_opt = 0, []
    tic = time.clock()
    model = KronRLsMKL(dataset)
    cmd = "Dataset:"+dataset+" CVS: "+str(cvs)+"\n"+str(model)
    print(cmd)
    aupr_vec, auc_vec = train(model, cv_data, X, D, T)
    aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
    auc_avg, auc_conf = mean_confidence_interval(auc_vec)
    print("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock()-tic))
    if auc_avg > max_auc:
        max_auc = auc_avg
        auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
    cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f\n" % (auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4])
    cmd +=" max auc:%.6f\n"%(max_auc)
    print(cmd)

def daspfind_cv_eval(method, dataset, cv_data, X, D, T, cvs, para):
    max_auc, auc_opt = 0, []
    tic = time.clock()
    model = DASPFIND()
    cmd = "Dataset:"+dataset+" CVS: "+str(cvs)+"\n"+str(model)
    print(cmd)
    aupr_vec, auc_vec = train(model, cv_data, X, D, T)
    aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
    auc_avg, auc_conf = mean_confidence_interval(auc_vec)
    print("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock()-tic))
    if auc_avg > max_auc:
        max_auc = auc_avg
        auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
    cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f\n" % (auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4])
    print(cmd)

def nrlmf_cv_eval(method, dataset, cv_data, X, D, T, cvs, para):
    max_auc, auc_opt = 0, []
    for r in [50, 100]:
        for x in np.arange(-5, 2):
            for y in np.arange(-5, 3):
                for z in np.arange(-5, 1):
                    for t in np.arange(-3, 1):
                        tic = time.clock()
                        model = NRLMF(cfix=para['c'], K1=para['K1'], K2=para['K2'], num_factors=r, lambda_d=2**np.int(x), lambda_t=2**np.int(x), alpha=2**np.int(y), beta=2**np.int(z), theta=2**np.int(t), max_iter=100)
                        cmd = "Dataset:"+dataset+" CVS: "+str(cvs)+"\n"+str(model)
                        print(cmd)
                        aupr_vec, auc_vec = train(model, cv_data, X, D, T)
                        aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
                        auc_avg, auc_conf = mean_confidence_interval(auc_vec)
                        print("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock()-tic))
                        if auc_avg > max_auc:
                            max_auc = auc_avg
                            auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
    cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f\n" % (auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4])
    print(cmd)


def netlaprls_cv_eval(method, dataset, cv_data, X, D, T, cvs, para):
    max_auc, auc_opt = 0, []
    for x in np.arange(-6, 3):  # [-6, 2]
        for y in np.arange(-6, 3):  # [-6, 2]
            tic = time.clock()
            model = NetLapRLS(gamma_d=10**(x), gamma_t=10**(x), beta_d=10**(y), beta_t=10**(y))
            cmd = "Dataset:"+dataset+" CVS: "+str(cvs)+"\n"+str(model)
            print(cmd)
            aupr_vec, auc_vec = train(model, cv_data, X, D, T)
            aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
            auc_avg, auc_conf = mean_confidence_interval(auc_vec)
            print("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock()-tic))
            if auc_avg > max_auc:
                max_auc = auc_avg
                auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
    cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f\n" % (auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4])
    print(cmd)


def blmnii_cv_eval(method, dataset, cv_data, X, D, T, cvs, para):
    max_auc, auc_opt = 0, []
    for x in np.arange(0, 1.1, 0.1):
        tic = time.clock()
        model = BLMNII(alpha=x, avg=False)
        cmd = "Dataset:"+dataset+" CVS: "+str(cvs)+"\n"+str(model)
        print(cmd)
        aupr_vec, auc_vec = train(model, cv_data, X, D, T)
        aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
        auc_avg, auc_conf = mean_confidence_interval(auc_vec)
        print("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock()-tic))
        if auc_avg > max_auc:
            max_auc = auc_avg
            auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
    cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f\n" % (auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4])
    print(cmd)


def wnngip_cv_eval(method, dataset, cv_data, X, D, T, cvs, para):
    max_auc, auc_opt = 0, []
    for x in np.arange(0.1, 1.1, 0.1):
        for y in np.arange(0.0, 1.1, 0.1):
            tic = time.clock()
            model = WNNGIP(T=x, sigma=1, alpha=y)
            cmd = "Dataset:"+dataset+" CVS: "+str(cvs)+"\n"+str(model)
            print(cmd)
            aupr_vec, auc_vec = train(model, cv_data, X, D, T)
            aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
            auc_avg, auc_conf = mean_confidence_interval(auc_vec)
            print("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock()-tic))
            if auc_avg > max_auc:
                max_auc = auc_avg
                auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
    cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f\n" % (auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4])
    print(cmd)
# def ndaf_cv_eval(method, dataset, cv_data, X, D, T, cvs, para):
#    max_auc, auc_opt = 0, []
#    tic = time.clock()
#    model = NDAF()
#    cmd = "Dataset:"+dataset+" CVS: "+str(cvs)+"\n"+str(model)
#    print(cmd)
#    aupr_vec, auc_vec = train(model, cv_data, X, D, T)
#    aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
#    auc_avg, auc_conf = mean_confidence_interval(auc_vec)
#    print("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock()-tic))
#    if auc_avg > max_auc:
#        max_auc = auc_avg
#        auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
#    cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
#    cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f\n" % (auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4])
#    print(cmd)
def grmf_cv_eval(method, dataset, cv_data, X, D, T, cvs, para):
   max_auc, auc_opt = 0, []
   tic = time.clock()
   model = GRMF(cvs)
   cmd = "Dataset:"+dataset+" CVS: "+str(cvs)+"\n"+str(model)
   print(cmd)
   aupr_vec, auc_vec = train(model, cv_data, X, D, T)
   aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
   auc_avg, auc_conf = mean_confidence_interval(auc_vec)
   print("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock()-tic))
   if auc_avg > max_auc:
       max_auc = auc_avg
       auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
   cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
   cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f\n" % (auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4])
   print(cmd)

def pudt_cv_eval(method, dataset, cv_data, X, D, T, cvs, para):
   max_auc, auc_opt = 0, []
   tic = time.clock()
   model = PUDT(dataset)
   cmd = "Dataset:"+dataset+" CVS: "+str(cvs)+"\n"+str(model)
   print(cmd)
   aupr_vec, auc_vec = train(model, cv_data, X, D, T)
   aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
   auc_avg, auc_conf = mean_confidence_interval(auc_vec)
   print("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock()-tic))
   if auc_avg > max_auc:
       max_auc = auc_avg
       auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
   cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
   cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f\n" % (auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4])
   print(cmd)

def netcbp_cv_eval(method, dataset, cv_data, X, D, T, cvs, para):
   max_auc, auc_opt = 0, []
   tic = time.clock()
   model = NetCBP()
   cmd = "Dataset:"+dataset+" CVS: "+str(cvs)+"\n"+str(model)
   print(cmd)
   aupr_vec, auc_vec = train(model, cv_data, X, D, T)
   aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
   auc_avg, auc_conf = mean_confidence_interval(auc_vec)

   print("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock()-tic))
   if auc_avg > max_auc:
       max_auc = auc_avg
       auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
   cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
   cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f\n" % (auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4])
   print(cmd)


def kbmf_cv_eval(method, dataset, cv_data, X, D, T, cvs, para):
   max_auc, auc_opt = 0, []
   for d in [50, 100]:
       tic = time.clock()
       model = KBMF(num_factors=d)
       cmd = "Dataset:"+dataset+" CVS: "+str(cvs)+"\n"+str(model)
       print(cmd)
       aupr_vec, auc_vec = train(model, cv_data, X, D, T)
       aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
       auc_avg, auc_conf = mean_confidence_interval(auc_vec)
       print("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock()-tic))
       if auc_avg > max_auc:
           max_auc = auc_avg
           auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
   cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
   cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f\n" % (auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4])
   print(cmd)


def cmf_cv_eval(method, dataset, cv_data, X, D, T, cvs, para):
    max_aupr, aupr_opt = 0, []
    for d in [50, 100]:
        for x in np.arange(-2, -1):
            for y in np.arange(-3, -2):
                for z in np.arange(-3, -2):
                    tic = time.clock()
                    model = CMF(K=d, lambda_l=2**np.int(x), lambda_d=2**np.int(y), lambda_t=2**np.int(z), max_iter=30)
                    cmd = "Dataset:"+dataset+" CVS: "+str(cvs)+"\n"+str(model)
                    print(cmd)
                    aupr_vec, auc_vec = train(model, cv_data, X, D, T)
                    aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
                    auc_avg, auc_conf = mean_confidence_interval(auc_vec)
                    print("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock()-tic))
                    if aupr_avg > max_aupr:
                        max_aupr = aupr_avg
                        aupr_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
    cmd = "Optimal parameter setting:\n%s\n" % aupr_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f\n" % (aupr_opt[1], aupr_opt[2], aupr_opt[3], aupr_opt[4])
    print(cmd)


def brdti_cv_eval( dataset, cv_data, X, D, T, cvs, para):
   # max_metric, metric_opt, optArg = 0, [], []
    max_aupr, aupr_opt = 0, []
    for d in [50, 100]:
        for lr in [0.1]:
            for glR in [0.01, 0.05, 0.1, 0.3]:
                for bR in [1]:
                    for caRU in [0.05, 0.1, 0.5, 0.9]:  # , 1.5
                        for caRI in [1]:
                            tic = time.clock()
                            ar = {
                                'D': d,
                                'learning_rate': lr,
                                'max_iters': 100,
                                'bias_regularization': bR,
                                'simple_predict': False,
                                'global_regularization': glR,
                                "cbSim": para["cbSim"],
                                'cb_alignment_regularization_user': caRU,
                                'cb_alignment_regularization_item': caRI}

                            model = BRDTI(ar)
                            cmd = "Dataset:" + dataset + " CVS: " + str(cvs) + "\n" + str(model)
                            print(cmd)

                            #aupr_vec, auc_vec, ndcg_vec, ndcg_inv_vec, results = train(model, cv_data, X, D, T)
                            aupr_vec, auc_vec = train(model, cv_data, X, D, T)
                            aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
                            auc_avg, auc_conf = mean_confidence_interval(auc_vec)
                           # ndcg_avg, ndcg_conf = mean_confidence_interval(ndcg_vec)
                           # ndcg_inv_avg, ndcg_inv_conf = mean_confidence_interval(ndcg_inv_vec)
                           # with open(os.path.join(output_dir, "optPar",
                           #                        "proc_" + dataset + "_" + str(cvs) + "_" + method + ".txt"),
                           #           "a") as procFile:
                           #     procFile.write(str(model) + ": ")
                           #     procFile.write("auc:%.6f, aupr: %.6f,ndcg: %.6f,ndcg_inv: %.6f, Time:%.6f\n" % (
                           #     #auc_avg, aupr_avg, ndcg_avg, ndcg_inv_avg, time.clock() - tic))
                           #     auc_avg, aupr_avg, time.clock() - tic))

                            print("auc:%.6f, aupr: %.6f,ndcg: %.6f,ndcg_inv: %.6f, Time:%.6f\n"
                                  #% (auc_avg, aupr_avg, ndcg_avg, ndcg_inv_avg, time.clock() - tic))
                                  % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock() - tic))
                            #metric = ndcg_inv_avg + ndcg_avg
                            #if metric > max_metric:
                            #    max_metric = metric
                            #    metric_opt = [cmd, auc_avg, aupr_avg, ndcg_avg, ndcg_inv_avg]
                            #    optArg = {"d": d, "lr": lr, "glR": glR, "caRU": caRU, "caRI": caRI}
                            #    # each time a better solution is found, the params are stored
                            #    with open(os.path.join(output_dir, "optPar",
                            #                           "res_" + dataset + "_" + str(cvs) + "_" + method + ".txt"),
                            #              "w") as resFile:
                            #        resFile.write(str(optArg) + "\n" + str(metric_opt))
                            if aupr_avg > max_aupr:
                                max_aupr = aupr_avg
                                aupr_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
    cmd = "Optimal parameter setting:\n%s\n" % aupr_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f\n" % (aupr_opt[1], aupr_opt[2], aupr_opt[3], aupr_opt[4])
    print(cmd)

def ddr_cv_eval(method, dataset, cv_data, X, D, T, cvs, para):
    max_auc, auc_opt = 0, []
    tic = time.clock()
    model = DDR(dataset,cvs)
    cmd = "Dataset:"+dataset+" CVS: "+str(cvs)+"\n"+str(model)
    print(cmd)
    aupr_vec, auc_vec = train(model, cv_data, X, D, T)
    aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
    auc_avg, auc_conf = mean_confidence_interval(auc_vec)
    print("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock()-tic))
    if auc_avg > max_auc:
        max_auc = auc_avg
        auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
    cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f\n" % (auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4])
    cmd +=" max auc:%.6f\n"%(max_auc)
    print(cmd)
