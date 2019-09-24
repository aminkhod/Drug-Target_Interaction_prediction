'''
[1] Yong Liu, Min Wu, Chunyan Miao, Peilin Zhao, Xiao-Li Li, "Neighborhood Regularized Logistic Matrix Factorization for Drug-target Interaction Prediction", under review.
'''
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc
from functions import *
import matplotlib.pyplot as plt

class DASPFIND:

    def __init__(self,alpha):
     self.alpha=alpha

    def fix_model(self, W, intMat, drugMat, targetMat,num, cvs, dataset, seed=None):
        self.dataset = dataset
        self.num = num
        self.cvs = cvs
        self.seed = seed

        self.intMat=intMat
        x, y = np.where(self.intMat > 0)
        self.train_drugs, self.train_targets = set(x.tolist()), set(y.tolist())
        self.dsMat = drugMat
        self.tsMat = targetMat
        # score = np.zeros((np.size(self.intMat[:,1]), np.size(self.intMat[1,:])))
        R = W * intMat
        MT = np.array(self.tsMat)
        MD = np.array(self.dsMat)
        MD2=np.power(MD, (self.alpha * 2))
        MT2=np.power(MT, (self.alpha * 2))
        MD3=np.power(MD, (self.alpha * 3))
        MT3=np.power(MT, (self.alpha * 3))
        buf1 = np.matmul(MD2,R)
        buf2=np.matmul(R,MT2)
        buf3 = np.matmul(np.dot(MD3,MD3),R)
        buf4=np.matmul(R,np.dot(MT3,MT3))
        buf5=np.matmul(np.dot(MD3,R),MT3)
        Rt=np.transpose(R)
        rtnodiag=np.dot(Rt,R)
        # g=np.diag(rtnodiag)
        for i in range(len(Rt)):
            rtnodiag[i,i] = 0
        buf6=np.matmul(R,rtnodiag)

        # buf6= np.dot(R,np.transpose(np.dot(R,np.transpose(R))))
        score=buf1+buf2+buf5+buf3 + buf4+buf5+buf6
        # for i in range(np.size(MD[1, :])):
        #     for j in range(np.size(MT[1, :])):
        #         if (R[i, j] != 1):
        #             for k in range(np.size(R[1, :])):
        #                 if (R[i, k] == 1):
        #                     lent = 2
        #                     Pw = MT[k, j] ** (self.alpha * lent)
        #                     score[i, j] += (Pw)
        #
        #             for z in range(np.size(R[:, 1])):
        #                 if (R[z, j] == 1):
        #                     lent = 2
        #                     Pw = MD[z, i] ** (self.alpha * lent)
        #                     score[i, j] += (Pw)
        # for k in range(np.size(R[1, :])):
        #     for z in range(np.size(R[:, 1])):
        #         if (R[z, k] == 1):
        #             for i in range(np.size(MD[1, :])):
        #                 for j in range(np.size(MT[1, :])):
        #                     if (R[i, j] != 1):
        #                         lent = 3
        #                         Pw = MT[k, j]
        #                         Pw *= MD[i, z]
        #                         Pw = Pw ** (self.alpha * lent)
        #                         score[i, j] += (Pw)
        self.predictR=score
    def predict_scores(self, test_data, N):
        dinx = np.array(list(self.train_drugs))
        DS = self.dsMat[:, dinx]
        tinx = np.array(list(self.train_targets))
        TS = self.tsMat[:, tinx]
        scores = []
        for d, t in test_data:
            if d in self.train_drugs:
                if t in self.train_targets:
                    val = np.sum(self.U[d, :]*self.V[t, :])
                else:
                    jj = np.argsort(TS[t, :])[::-1][:N]
                    val = np.sum(self.U[d, :]*np.dot(TS[t, jj], self.V[tinx[jj], :]))/np.sum(TS[t, jj])
            else:
                if t in self.train_targets:
                    ii = np.argsort(DS[d, :])[::-1][:N]
                    val = np.sum(np.dot(DS[d, ii], self.U[dinx[ii], :])*self.V[t, :])/np.sum(DS[d, ii])
                else:
                    ii = np.argsort(DS[d, :])[::-1][:N]
                    jj = np.argsort(TS[t, :])[::-1][:N]
                    v1 = DS[d, ii].dot(self.U[dinx[ii], :])/np.sum(DS[d, ii])
                    v2 = TS[t, jj].dot(self.V[tinx[jj], :])/np.sum(TS[t, jj])
                    val = np.sum(v1*v2)
            scores.append(np.exp(val)/(1+np.exp(val)))
        return np.array(scores)

    def evaluation(self, test_data, test_label):
        score = self.predictR
        import pandas as pd
        score = pd.DataFrame(score)
        score.to_csv('../data/datasets/EnsambleDTI/daspfind_'+str(self.dataset)+'_s'+
                      str(self.cvs)+'_'+str(self.seed)+'_'+str(self.num)+'.csv', index=False)
        score = self.predictR[test_data[:, 0], test_data[:, 1]]
        prec, rec, thr = precision_recall_curve(test_label,np.array(score))
        aupr_val = auc(rec, prec)
        fpr, tpr, thr = roc_curve( test_label,np.array(score))
        auc_val = auc(fpr, tpr)
        print("AUPR: " + str(aupr_val) + ", AUC: " + str(auc_val))

        return aupr_val, auc_val

    def __str__(self):
        return "Model: DASPFIND"
