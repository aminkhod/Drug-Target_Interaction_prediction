import os
import numpy as np

'''
Pymatbridge communicates with Matlab using zeromq. 
So before installing pymatbridge you must have zmq library and pyzmq installed on your machine. 
If you intend to use the Matlab magic extension, you'll also need IPython. 
To make pymatbridge work properly, please follow the following steps.
'''
from pymatbridge import Matlab
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

class NetCBP:

    def __init__(self,alpha=0.2,beta=0.2):
        self.alpha=alpha
        self.beta=beta

    def fix_model(self, W, intMat, drugMat, targetMat,num, cvs, dataset, seed=None):
        self.dataset = dataset
        self.num = num
        self.cvs = cvs
        self.seed = seed
        R = W * intMat
        drugMat = (drugMat + drugMat.T) / 2
        targetMat = (targetMat + targetMat.T) / 2
        mlab = Matlab()
        mlab.start()
        res = mlab.run_func(os.path.realpath(os.sep.join(["netcbp","NetCBP.m"])),{'Kx': drugMat, 'Kz': targetMat, 'Y': R, 'alpha': self.alpha, 'beta': self.beta})
        self.predictR = res['result']
       
        mlab.stop()

    def predict_scores(self, test_data, N):
        inx = np.array(test_data)
        return self.predictR[inx[:, 0], inx[:, 1]]

    def evaluation(self, test_data, test_label):
        score = self.predictR
        import pandas as pd
        score = pd.DataFrame(score)
        score.to_csv('../data/datasets/EnsambleDTI/netcbp_'+str(self.dataset)+'_s'+
                      str(self.cvs)+'_'+str(self.seed)+'_'+str(self.num)+'.csv', index=False)
        score = self.predictR[test_data[:, 0], test_data[:, 1]]
        prec, rec, thr = precision_recall_curve(test_label, np.array(score))
        aupr_val = auc(rec, prec)
        # plt.step(rec, prec, color='b', alpha=0.2,
        #          where='post')
        # plt.fill_between(rec, prec, step='post', alpha=0.2,
        #                  color='b')
        #
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.ylim([0.0, 1.05])
        # plt.xlim([0.0, 1.0])
        # plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
        #     average_prec))

        fpr, tpr, thr = roc_curve(test_label, np.array(score))
        auc_val = auc(fpr, tpr)
        print("AUPR: " + str(aupr_val) + ", AUC: " + str(auc_val))

        return aupr_val, auc_val

    def __str__(self):
        return "Model: NetCBP"
