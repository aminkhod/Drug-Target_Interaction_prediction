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

class NDAF:

    def __init__(self):
        1

    def fix_model(self, W, intMat, drugMat, targetMat, seed=None):
        R = W * intMat
        drugMat = (drugMat + drugMat.T) / 2
        targetMat = (targetMat + targetMat.T) / 2
        mlab = Matlab()
        mlab.start()
        # print os.getcwd()
        # self.predictR = mlab.run_func(os.sep.join([os.getcwd(), "kbmf2k", "kbmf.m"]), {'Kx': drugMat, 'Kz': targetMat, 'Y': R, 'R': self.num_factors})['result']
        res = mlab.run_func(os.path.realpath(os.sep.join(['ndaf',"runNDAF.m"])),{'Kx': drugMat, 'Kz': targetMat, 'Y': R})
        self.predictR = res['result']
        # print os.path.realpath(os.sep.join(['../kbmf2k', "kbmf.m"]))
        mlab.stop()

    def predict_scores(self, test_data, N):
        inx = np.array(test_data)
        return self.predictR[inx[:, 0], inx[:, 1]]

    def evaluation(self, test_data, test_label):
        score = self.predictR[test_data[:, 0], test_data[:, 1]]
        average_prec = average_precision_score(test_label, np.array(score))
        prec, rec, thr = precision_recall_curve(test_label, np.array(score))
        aupr_val = auc(rec, prec)
        plt.step(rec, prec, color='b', alpha=0.2,
                 where='post')
        plt.fill_between(rec, prec, step='post', alpha=0.2,
                         color='b')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
            average_prec))
        fpr, tpr, thr = roc_curve(test_label, np.array(score))
        auc_val = auc(fpr, tpr)
        print("AUPR: " + str(aupr_val) + ", AUC: " + str(auc_val))

        return aupr_val, auc_val

    def __str__(self):
        return "Model: NDAF"
