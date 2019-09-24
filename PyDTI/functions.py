
import os
import numpy as np
from collections import defaultdict
import rank_metrics as rank
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.csgraph import minimum_spanning_tree


def load_data_from_file(dataset, folder):
    with open(os.path.join(folder, dataset+"_admat_dgc.txt"), "r") as inf:
        inf.readline()
        int_array = [line.strip("\n").split()[1:] for line in inf]
    inf.close()
    with open(os.path.join(folder, dataset+"_simmat_dc.txt"), "r") as inf:  # the drug similarity file
        inf.readline()
        drug_sim = [line.strip("\n").split()[1:] for line in inf]
    inf.close()
    with open(os.path.join(folder, dataset+"_simmat_dg.txt"), "r") as inf:  # the target similarity file
        inf.readline()
        target_sim = [line.strip("\n").split()[1:] for line in inf]
    inf.close()
    intMat = np.array(int_array, dtype=np.float64).T    # drug-target interaction matrix
    drugMat = np.array(drug_sim, dtype=np.float64)      # drug similarity matrix
    targetMat = np.array(target_sim, dtype=np.float64)  # target similarity matrix
    return intMat, drugMat, targetMat


def get_drugs_targets_names(dataset, folder):
    with open(os.path.join(folder, dataset+"_admat_dgc.txt"), "r") as inf:
        drugs = inf.readline().strip("\n").split()
        targets = [line.strip("\n").split()[0] for line in inf]
    inf.close()
    return drugs, targets

def cross_validation(intMat, seeds, cv=0, invert=0, num=10):
    cv_data = defaultdict(list)
    for seed in seeds:
        num_drugs, num_targets = intMat.shape
        prng = np.random.RandomState(seed)
        if cv == 0:
            index = prng.permutation(num_drugs)
            step = int(index.size / num)
            for i in range(num):
                if i < num - 1:
                    ii = index[i * step:(i + 1) * step]
                else:
                    ii = index[i * step:]
                test_data = np.array([[k, j] for k in ii for j in range(num_targets)], dtype=np.int32)
                x, y = test_data[:, 0], test_data[:, 1]
                test_label = intMat[x, y]
                W = np.ones(intMat.shape)
                W[x, y] = 0
                if invert:
                    W_T = W.T
                    test_data_T = np.column_stack((y, x))
                    cv_data[seed].append((W_T, test_data_T, test_label))
                else:
                    cv_data[seed].append((W, test_data, test_label))
        if cv == 1:
            index = prng.permutation(intMat.size)
            step = int(index.size / num)
            for i in range(num):
                if i < num - 1:

                    ii = index[i * step:(i + 1) * step]
                else:
                    ii = index[i * step:]
                test_data = np.array([[k / num_targets, k % num_targets] for k in ii], dtype=np.int32)
                x, y = test_data[:, 0], test_data[:, 1]
                test_label = intMat[x, y]
                W = np.ones(intMat.shape)
                W[x, y] = 0
                if invert:
                    W_T = W.T
                    test_data_T = np.column_stack((y, x))
                    cv_data[seed].append((W_T, test_data_T, test_label))
                else:
                    cv_data[seed].append((W, test_data, test_label))
        if cv == 2:
            n, m = intMat.shape[0], intMat.shape[1]
            adjMat = np.bmat([[np.zeros((n, n)), intMat], [intMat.T, np.zeros((m, m))]])
            graphAdj = csr_matrix(adjMat)
            mst = minimum_spanning_tree(graphAdj)
            mstArray = mst.toarray()
            mstArray2_Final = np.logical_or(mstArray, mstArray.T).astype(int)
            testMat = (adjMat - mstArray2_Final)[0:n, n:n + m]
            p, q = np.where(intMat == 0)
            x, y = np.where(testMat == 1)
            xy_zip = list(zip(x, y))
            pq_zip = list(zip(p, q))
            # pq_xy_zip = set().union(xy_zip, pq_zip)
            testIndices_pq = [(i * m + j) for (i, j) in pq_zip]
            testIndices_xy = [(i * m + j) for (i, j) in xy_zip]
            index_pq = np.random.permutation(testIndices_pq)
            index_xy = np.random.permutation(testIndices_xy)
            step_pq = int(index_pq.size / num)
            step_xy = int(index_xy.size / num)
            for i in range(num):
                if i < num - 1:
                    ii_pq = index_pq[i * step_pq:(i + 1) * step_pq]
                    ii_xy = index_xy[i * step_xy:(i + 1) * step_xy]
                else:
                    ii_pq = index_pq[i * step_pq:]
                    ii_xy = index_xy[i * step_xy:]
                test_data_xy = np.array([[k / num_targets, k % num_targets] for k in ii_xy], dtype=np.int32)
                test_data_pq = np.array([[k / num_targets, k % num_targets] for k in ii_pq], dtype=np.int32)
                test_data = np.concatenate([test_data_xy, test_data_pq])
                x, y = test_data[:, 0], test_data[:, 1]
                test_label = intMat[x, y]
                W = np.ones(intMat.shape)
                W[x, y] = 0

                if invert:
                    W_T = W.T
                    test_data_T = np.column_stack((y,x))
                    cv_data[seed].append((W_T, test_data_T, test_label))
                else:
                    cv_data[seed].append((W, test_data, test_label))

    return cv_data
'''
def cross_validation1(intMat, seeds, cv=0, invert=0, num=10):
    cv_data = defaultdict(list)
    for seed in seeds:
        num_drugs, num_targets = intMat.shape
        prng = np.random.RandomState(seed)
        if cv == 0:
            index = prng.permutation(num_drugs)
        if cv == 1:
            index = prng.permutation(intMat.size)
        step = int(index.size/num)
        for i in range(num):
            if i < num-1:
                ii =index[i*step:(i+1)*step]
            else:
                ii = index[i*step:]
            if cv == 0:
                test_data = np.array([[k, j] for k in ii for j in range(num_targets)], dtype=np.int32)
            elif cv == 1:
                test_data = np.array([[k/num_targets, k % num_targets] for k in ii], dtype=np.int32)
            x, y = test_data[:, 0], test_data[:, 1]
            test_label = intMat[x, y]
            W = np.ones(intMat.shape)
            W[x, y] = 0
            if invert:
                W_T = W.T
                test_data_T = np.column_stack((y,x))
                cv_data[seed].append((W_T, test_data_T, test_label))
            else:
                cv_data[seed].append((W, test_data, test_label))

    return cv_data
'''

def train(model, cv_data, intMat, drugMat, targetMat,cvs, dataset):
    # aupr, auc , prec, rec, thr, name = [], [] ,[], [] , [], []
    aupr, auc = [], []
    for seed in cv_data.keys():
        num=1
        for W, test_data, test_label in cv_data[seed]:
            # print(str(model)[7:18])
            # if (num<4) :
            if str(model)[7:18] == 'EnsambleDTI':
                model.fix_model(W, intMat, drugMat, targetMat, test_data, test_label, num, cvs, dataset, seed)
            else:
                model.fix_model(W, intMat, drugMat, targetMat, seed)
            # prec, rec, thr, name,aupr_val, auc_val = model.evaluation(test_data, test_label)
            aupr_val, auc_val = model.evaluation(test_data, test_label)
            aupr.append(aupr_val)
            auc.append(auc_val)
            # print("")
            num +=1
            # plot_aupr(prec , rec ,thr , name)
    return np.array(aupr, dtype=np.float64), np.array(auc, dtype=np.float64)
def svd_init(M, num_factors):
    from scipy.linalg import svd
    U, s, V = svd(M, full_matrices=False)
    ii = np.argsort(s)[::-1][:num_factors]
    s1 = np.sqrt(np.diag(s[ii]))
    U0, V0 = U[:, ii].dot(s1), s1.dot(V[ii, :])
    return U0, V0.T


def mean_confidence_interval(data, confidence=0.95):
    import scipy as sp
    import scipy.stats
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, h


def write_metric_vector_to_file(auc_vec, file_name):
    np.savetxt(file_name, auc_vec, fmt='%.6f')


def load_metric_vector(file_name):
    return np.loadtxt(file_name, dtype=np.float64)


def plot_aupr( prec, rec, thr, name):
    import matplotlib.pyplot as plt
    plt.clf()
    plt.ioff()
    plt.plot(rec, prec, label='Precision-Recall')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall ')
    plt.legend(loc="lower left")
    fig = plt.figure()
    fig.savefig(name + '.png', bbox_inches='tight')
    fig.savefig(name + '.pdf', bbox_inches='tight')


def normalized_discounted_cummulative_gain(test_data, test_label, scores):
    unique_users = np.unique(test_data[:, 0])
    user_array = test_data[:, 0]
    ndcg = []

    for u in unique_users:
        indices_u = np.in1d(user_array, [u])
        labels_u = test_label[indices_u].astype(float)
        scores_u = scores[indices_u].astype(float)
        # ndcg is calculated only for the users with some positive examples
        if not all(i <= 0.001 for i in labels_u):
            tmp = np.c_[labels_u, scores_u]
            tmp = tmp[tmp[:, 1].argsort()[::-1], :]
            ordered_labels = tmp[:, 0]
            ndcg_u = rank.ndcg_at_k(ordered_labels, ordered_labels.shape[0], 1)
            ndcg.append(ndcg_u)
    return np.mean(ndcg)


