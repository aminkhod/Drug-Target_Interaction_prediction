import subprocess
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc


class DTHYBRID:
    def __init__(self,dataset):
        self.dataset=dataset

    def fix_model(self, W, intMat, drugMat, targetMat, seed):
        np.savetxt('DTHybridW.txt', W)
        command = 'Rscript'
        path2script = 'DTHybrid.R'
        args=[self.dataset]
        cmd = [command, path2script] + args
        subprocess.call(cmd,universal_newlines=False)
        #self.predictR =  open('DTHybridscore.txt','w')


    def predict_scores(self, test_data, N):
        inx = np.array(test_data)
        return self.predictR[inx[:, 0], inx[:, 1]]

    def evaluation(self, test_data, test_label):
        with open("DTHybridscore.txt", "r") as inf:
            inf.readline()
            int_array = [line.strip("\n").split()[:] for line in inf]
        score=np.array(int_array, dtype=np.float64)
        scores=score[test_data[:, 0],test_data[:, 1]]
        prec, rec, thr = precision_recall_curve(test_label, np.array(scores))
        aupr_val = auc(rec, prec)
        fpr, tpr, thr = roc_curve(test_label, np.array(scores))
        auc_val = auc(fpr, tpr)
        print("AUPR: " + str(aupr_val) + ", AUC: " + str(auc_val))
        return aupr_val, auc_val

    def __str__(self):
        return "Model: DTHybrid"
