import sys
import os
import numpy as np
from pymatbridge import Matlab
from sklearn.metrics import precision_recall_curve, roc_curve,roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

with open("exp.txt", "r") as inf:
    inf.readline()
    int_array = [line.strip("\n").split()[:] for line in inf]


predictR=np.array(int_array, dtype=np.float64)

with open("exp1.txt", "r") as inf1:
    inf1.readline()
    int_array1 = [line.strip("\n").split()[:] for line in inf1]
final_lable = np.array(int_array1, dtype=np.float64)

mlab = Matlab()
mlab.start()
# print os.getcwd()
# self.predictR = mlab.run_func(os.sep.join([os.getcwd(), "kbmf2k", "kbmf.m"]), {'Kx': drugMat, 'Kz': targetMat, 'Y': R, 'R': self.num_factors})['result']
res = mlab.run_func(os.path.realpath(os.sep.join(["..\pudt", "AUC.m"])),
                    {'test_targets': final_lable , 'output': predictR})
predictAUC = res['result']
# print os.path.realpath(os.sep.join(['../kbmf2k', "kbmf.m"]))
mlab.stop()
# score = self.predictR[test_data[:, 0], test_data[:, 1]]
score = predictAUC
fpr, tpr, thr = roc_curve(final_lable, np.array(predictR))
auc_val = auc(fpr, tpr)

print("predictAUC: " + str(predictAUC) + ", AUC: " + str(auc_val))