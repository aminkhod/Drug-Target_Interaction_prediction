from Graph_utils import *
from functions import cross_validation,mean_confidence_interval
from Classify import *
from SNF import *
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc


class DDR:
    def __init__(self, dataset,cv):
        self.dataset = dataset
        self.cv=cv

    def get_features_per_fold(self, R_train, D_sim, T_sim, pair):
        accum_DDD, max_DDD = get_two_hop_similarities(D_sim, D_sim)
        accum_TTT, max_TTT = get_two_hop_similarities(T_sim, T_sim)

        accum_DDT, max_DDT = get_drug_relation(D_sim, R_train)
        accum_DTT, max_DTT = get_relation_target(T_sim, R_train)

        accum_DDDT, _ = get_drug_relation(accum_DDD, R_train)
        _, max_DDDT = get_drug_relation(max_DDD, R_train)

        accum_DTTT, _ = get_relation_target(accum_TTT, R_train)
        _, max_DTTT = get_relation_target(max_TTT, R_train)

        accum_DTDT, max_DTDT = get_DTDT(R_train)

        accum_DDTT, max_DDTT = get_DDTT(R_train, D_sim, T_sim)

        features = []

        features.append(mat2vec(accum_DDT))
        features.append(mat2vec(max_DDT))
        features.append(mat2vec(accum_DTT))
        features.append(mat2vec(max_DTT))

        features.append(mat2vec(accum_DDDT))

        features.append(mat2vec(max_DDDT))

        if pair:
            features.append(mat2vec(accum_DTDT))
            features.append(mat2vec(max_DTDT))

        features.append(mat2vec(accum_DTTT))

        features.append(mat2vec(max_DTTT))

        features.append(mat2vec(accum_DDTT))

        features.append(mat2vec(max_DDTT))

        return features

    def get_similarities(self,sim_file, dMap):
        sim = []

        for line in open(sim_file).readlines():
            edge_list = get_edge_list(line.strip())
            sim.append(make_sim_matrix(edge_list, dMap))

        return sim

    def fix_model(self, W, intMat, drugMat, targetMat,num, cvs, dataset, seed=None):
        self.dataset = dataset
        self.num = num
        self.cvs = cvs
        self.seed = seed
        # if (self.cv >= 3):
        #     intMat = np.transpose(intMat)
        #     W = np.transpose(W)
        self.intMat=intMat
        R_all_train_test = "../data/datasets/DDR/"+self.dataset+"_admat_dgc_mat_2_line.txt"

        D_sim_file = "../data/datasets/DDR/"+self.dataset+"_D_SimLine_files.txt"
        T_sim_file = "../data/datasets/DDR/"+self.dataset+"_T_SimLine_files.txt"

        (D, T, DT_signature, aAllPossiblePairs, dDs, dTs, diDs, diTs) = get_All_D_T_thier_Labels_Signatures(
            R_all_train_test)

        # R = get_edge_list(R_all_train_test)
        # DT = get_adj_matrix_from_relation(R, dDs, dTs)


        D_sim = DDR.get_similarities(self,D_sim_file, dDs)
        T_sim = DDR.get_similarities(self,T_sim_file, dTs)

        row, col = intMat.shape

        all_matrix_index = []

        for i in range(row):
            for j in range(col):
                all_matrix_index.append([i, j])

        # for mode in ["p", "D", "T"]:
        if(self.cv==1):
            mode="p"
            pair = True
        elif(self.cv==2):
            mode="D"
            pair = False
        elif (self.cv == 3):
            mode = "T"
            pair = False
        elif (self.cv==4):
            mode = "p"
            pair = True
        # seeds = [7771, 8367, 22, 1812, 4659]

        # if mode == "p":
        #     cv_data = cross_validation(DT, seeds, 1, 10)
        #     pair = True
        # elif mode == "D":
        #     cv_data = cross_validation(DT, seeds, 0, 10)
        #     pair = False
        # elif mode == "T":
        #     pair = False
        #     cv_data = cross_validation(np.transpose(DT), seeds, 0, 10)

        self.labels = mat2vec(intMat)

        test_idx = []
        trails_AUPRs = []
        trails_AUCs = []
        trails_recall = []
        trails_precision = []
        trails_f1 = []

        # for seed in seeds:

        total = 0
        folds_features = []
        # for fold in cv_data[seed]:
        print("a")

        # if mode == "T":
        #     R_train = mask_matrix(DT, fold[1], True)
        # else:
        #     R_train = mask_matrix(DT, fold[1])  # by default transpose is false
        R_train= W * intMat
        DT_impute_D = impute_zeros(R_train, D_sim[0])
        DT_impute_T = impute_zeros(np.transpose(R_train), T_sim[0])

        GIP_D = Get_GIP_profile(np.transpose(DT_impute_D), "d")
        GIP_T = Get_GIP_profile(DT_impute_T, "t")

        WD = []
        WT = []

        for s in D_sim:
            WD.append(s)
        WD.append(GIP_D)

        for s in T_sim:
            WT.append(s)
        WT.append(GIP_T)
        D_SNF = SNF(WD, 3, 2)
        T_SNF = SNF(WT, 3, 2)

        DS_D = FindDominantSet(D_SNF, 5)
        DS_T = FindDominantSet(T_SNF, 5)

        np.fill_diagonal(DS_D, 0)
        np.fill_diagonal(DS_T, 0)

        features = get_features_per_fold(R_train, DS_D, DS_T, pair)
        folds_features.append(list(zip(*features)))

        # if mode == "T":
        #     test_idx.append([j * col + i for (i, j) in fold[1]])
        # else:
        #     test_idx.append([i * col + j for (i, j) in fold[1]])
        #
        # # print total
        # results = run_classification(folds_features, labels, test_idx)
        # trails_AUPRs.extend(results[2])
        # aupr, c1 = mean_confidence_interval(trails_AUPRs)
        # print("################Results###################")
        # print(("Mode: %s" % mode))
        # print(("Average AUPR: %f" % aupr))
        # print("###########################################")
        self.mode=mode
        self.features = features
        self.folds_features = folds_features

    def predict_scores(self, test_data, N):
        inx = np.array(test_data)
        return self.predictR[inx[:, 0], inx[:, 1]]

    def evaluation(self, test_data, test_label):
        row, col = self.intMat.shape
        # row, col = intMat.shape
        test_idx=[]
        test_idx.append([i * col + j for (i, j) in test_data])
        learning_rate = [0.01]  # [0.01,0.02,0.03,0.04,0.05,0.05,0.07,0.08,0.09,0.01]
        max_depth = [3, 4, 5, 6]

        # no_trees = [5,10,15]#range(5,62,2)
        no_trees = [100, 200, 300, 400]
        criterion = ["gini", "entropy"]
        auprres=[]
        aucres=[]
        scoretrain = []
        for cw in learning_rate:
        # cw =0.01
            for c in criterion:
        # c="gini"
                for t in no_trees:
        # t=100
                    scoreTesting , scoreTraining = run_classification_configuration(self.folds_features, self.labels, test_idx,t,cw,c,performance=False)
                    score = np.transpose(scoreTesting)
                    prec, rec, thr = precision_recall_curve(test_label, score)
                    # prec=np.array(prec[0:len(prec)])
                    # rec=np.array(rec[0:len(rec)])
                    aupr_val = auc(rec, prec)
                    fpr, tpr, thr = roc_curve(test_label, score)
                    auc_val = auc(np.array(fpr), np.array(tpr))
                    # print 'no_trees',t, 'max_depth',c, 'learning_rate',cw
                    # print 'total_AUPR_training:', round(AUPR_train,2)
                    # print 'total_AUPR_testing:', round(AUPR_test, 2)
                    # result.append(parameter_result)
                    auprres.append(aupr_val)
                    aucres.append(auc_val)
                    auprres.sort()
                    aucres.sort()
                    if aupr_val>= auprres[0]:
                        import pandas as pd
                        scoretrain = pd.DataFrame(scoreTraining)
        auc_val = aucres[0]
        aupr_val = auprres[0]

        scoretrain.to_csv('../data/datasets/EnsambleDTI/ddr_' + str(self.dataset) + '_s' +
                     str(self.cvs) + '_' + str(self.seed) + '_' + str(self.num) + '.csv', index=False)
        # score = self.predictR[test_data[:, 0], test_data[:, 1]]
        # prec, rec, thr = precision_recall_curve(test_label,np.array(score))
        # aupr_val = auc(rec, prec)
        # fpr, tpr, thr = roc_curve( test_label,np.array(score))
        # auc_val = auc(fpr, tpr)
        print("AUPR: " + str(aupr_val) + ", AUC: " + str(auc_val))
        return aupr_val, auc_val

    def __str__(self):
        return "Model: DDR"