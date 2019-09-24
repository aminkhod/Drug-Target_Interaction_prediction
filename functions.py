
import os
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.csgraph import minimum_spanning_tree


def load_data_from_file(dataset, folder):
    with open(os.path.join(folder, "protein_locals.txt"), "r") as inf:
        inf.next()
        protein_local_array = [line.strip("\n").split()[1:] for line in inf]

    with open(os.path.join(folder, "ppsim_align.txt"), "r") as inf:  # the protein similarity file
        inf.next()
        proteins_sim_align = [line.strip("\n").split()[1:] for line in inf]

    with open(os.path.join(folder, "ppsim_pssm.txt"), "r") as inf:  # the drug similarity file
        inf.next()
        proteins_sim_pssm = [line.strip("\n").split()[1:] for line in inf]

    with open(os.path.join(folder, "ppsim_string.txt"), "r") as inf:  # the drug similarity file
        inf.next()
        proteins_sim_string = [line.strip("\n").split()[1:] for line in inf]

    with open(os.path.join(folder, "ppsim_mf.txt"), "r") as inf:
        inf.next()
        proteins_sim_mf = [line.strip("\n").split()[1:] for line in inf]

    with open(os.path.join(folder, "ppsim_cc.txt"), "r") as inf:
        inf.next()
        proteins_sim_cc = [line.strip("\n").split()[1:] for line in inf]

    with open(os.path.join(folder, "ppsim_bp.txt"), "r") as inf:
        inf.next()
        proteins_sim_bp = [line.strip("\n").split()[1:] for line in inf]



    #///////////// JIMMI
    with open(os.path.join(folder, "_simmat_dg.txt"), "r") as inf:  # the target similarity file
        inf.next()
        location_sim = [line.strip("\n").split()[1:] for line in inf]
    #///////////////////////////////
    protein_local_Mat = np.array(protein_local_array, dtype=np.float64).T    # drug-target interaction matrix
    protein_Mat_align = np.array(proteins_sim_align, dtype=np.float64)      # drug similarity matrix
    protein_Mat_pssm = np.array(proteins_sim_pssm, dtype=np.float64)      # drug similarity matrix
    protein_Mat_string = np.array(proteins_sim_string, dtype=np.float64)      # drug similarity matrix
    protein_Mat_mf = np.array(proteins_sim_mf, dtype=np.float64)      # drug similarity matrix
    protein_Mat_cc = np.array(proteins_sim_cc, dtype=np.float64)      # drug similarity matrix
    protein_Mat_bp = np.array(proteins_sim_bp, dtype=np.float64)      # drug similarity matrix
    locationMat = np.array(location_sim, dtype=np.float64)  # target similarity matrix
    return protein_local_Mat, protein_Mat_align, protein_Mat_pssm, protein_Mat_string, protein_Mat_mf, protein_Mat_cc, protein_Mat_bp, locationMat


def get_drugs_targets_names(dataset, folder):
    with open(os.path.join(folder, dataset+"_admat_dgc.txt"), "r") as inf:
        drugs = inf.next().strip("\n").split()
        targets = [line.strip("\n").split()[0] for line in inf]
    return drugs, targets


def cross_validation(intMat, seeds, cv=0, num=10):
    cv_data = defaultdict(list)
    for seed in seeds:
        num_drugs, num_targets = intMat.shape
        prng = np.random.RandomState(seed)
        if cv == 0:
            index = prng.permutation(num_drugs)
            step = index.size/num
            for i in xrange(num):
                if i < num-1:
                    ii = index[i*step:(i+1)*step]
                else:
                    ii = index[i*step:]
                if cv == 0:
                    test_data = np.array([[k, j] for k in ii for j in xrange(num_targets)], dtype=np.int32)
                elif cv == 1:
                    test_data = np.array([[k/num_targets, k % num_targets] for k in ii], dtype=np.int32)
                elif cv ==2:
                    test_data = np.array([[k/num_targets, k % num_targets] for k in ii], dtype=np.int32)
                x, y = test_data[:, 0], test_data[:, 1]
                test_label = intMat[x, y]
                W = np.ones(intMat.shape)
                W[x, y] = 0
                cv_data[seed].append((W, test_data, test_label))
        if cv == 1:
            index = prng.permutation(intMat.size)
            step = index.size/num
            for i in xrange(num):
                if i < num-1:
                    ii = index[i*step:(i+1)*step]
                else:
                    ii = index[i*step:]
                if cv == 0:
                    test_data = np.array([[k, j] for k in ii for j in xrange(num_targets)], dtype=np.int32)
                elif cv == 1:
                    test_data = np.array([[k/num_targets, k % num_targets] for k in ii], dtype=np.int32)
                elif cv ==2:
                    test_data = np.array([[k/num_targets, k % num_targets] for k in ii], dtype=np.int32)
                x, y = test_data[:, 0], test_data[:, 1]
                test_label = intMat[x, y]
                W = np.ones(intMat.shape)
                W[x, y] = 0
                cv_data[seed].append((W, test_data, test_label))
        if cv == 2:
            n, m = intMat.shape[0], intMat.shape[1]
            adjMat = np.bmat([[np.zeros((n, n)), intMat], [intMat.T, np.zeros((m, m))]])
            graphAdj = csr_matrix(adjMat)
            mst = minimum_spanning_tree(graphAdj)
            mstArray = mst.toarray()
            mstArray2_Final = np.logical_or(mstArray, mstArray.T).astype(int)
            testMat = (adjMat - mstArray2_Final)[0:n, n:n+m]
            p, q = np.where(intMat == 0)
            x, y = np.where(testMat == 1)
            xy_zip = zip(x, y)
            pq_zip = zip (p, q)
            # pq_xy_zip = set().union(xy_zip, pq_zip)
            testIndices_pq = [(i*m + j) for (i, j) in pq_zip]
            testIndices_xy = [(i * m + j) for (i, j) in xy_zip]
            index_pq = np.random.permutation(testIndices_pq)
            index_xy = np.random.permutation(testIndices_xy)
            step_pq = index_pq.size / num
            step_xy = index_xy.size / num
            for i in xrange(num):
                if i < num-1:
                    ii_pq = index_pq[i*step_pq:(i+1)*step_pq]
                    ii_xy = index_xy[i * step_xy:(i + 1) * step_xy]
                else:
                    ii_pq = index_pq[i*step_pq:]
                    ii_xy = index_xy[i * step_xy:]
                test_data_xy = np.array([[k/num_targets, k % num_targets] for k in ii_xy], dtype=np.int32)
                test_data_pq = np.array([[k / num_targets, k % num_targets] for k in ii_pq], dtype=np.int32)
                test_data = np.concatenate([test_data_xy, test_data_pq])
                x, y = test_data[:, 0], test_data[:, 1]
                test_label = intMat[x, y]
                W = np.ones(intMat.shape)
                W[x, y] = 0
                cv_data[seed].append((W, test_data, test_label))

    return cv_data


def train(model, cv_data, intMat, drugMat, targetMat):
    aupr, auc = [], []
    for seed in cv_data.keys():
        for W, test_data, test_label in cv_data[seed]:
            model.fix_model(W, intMat, drugMat, targetMat, seed)
            aupr_val, auc_val = model.evaluation(test_data, test_label)
            aupr.append(aupr_val)
            auc.append(auc_val)
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
