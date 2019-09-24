
import edward as ed
# just tensorflow of 1.7.0!
import tensorflow as tf
from edward.models import  Normal, MultivariateNormalTriL, TransformedDistribution, NormalWithSoftplusScale
from edward.models.random_variable import RandomVariable

from tensorflow.contrib.distributions import Distribution
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, auc

class VBMKLMF:
    def __init__(self, dataset,cvs):

        self.dataset=dataset
        self.cvs=cvs


    def fix_model(self, W, intMat, drugMat, targetMat,num, cvs, dataset, seed=None):
        self.dataset = dataset
        self.num = num
        self.cvs = cvs
        self.seed = seed
        #self.intMat=intMat
        #x, y = np.where(self.intMat > 0)
        #self.train_drugs, self.train_targets = set(x.tolist()), set(y.tolist())
        #self.dsMat = drugMat
        #self.tsMat = targetMat
        # score = np.zeros((np.size(self.intMat[:,1]), np.size(self.intMat[1,:])))


        # Load interaction matrix
        admat = "../data/datasets/"+self.dataset+"_admat_dgc.txt"
        with open(admat) as f:
            ncols = len(f.readline().split('\t'))
        R_ = np.loadtxt(admat, skiprows=1, usecols=range(1, ncols), delimiter='\t', dtype=np.float64)
        I, J = R_.shape
        W=np.transpose(W)
        R_=W*R_

        # Load similarity matrices
        simmat_u = [
            "../data/datasets/"+self.dataset+"_simmat_dg.txt"]
        Ku = np.array([np.loadtxt(mat, skiprows=1, usecols=range(1, I + 1), delimiter='\t', dtype=np.float64) for mat in
                       simmat_u])

        simmat_v = ["../data/datasets/"+self.dataset+"_simmat_dc.txt",
                    "../data/datasets/"+self.dataset+"_simmat_dc_maccs_rbf.txt",
                    "../data/datasets/"+self.dataset+"_simmat_dc_maccs_tanimoto.txt",
                    "../data/datasets/"+self.dataset+"_simmat_dc_morgan_rbf.txt",
                    "../data/datasets/"+self.dataset+"_simmat_dc_morgan_tanimoto.txt"]
        Kv = np.array([np.loadtxt(mat, skiprows=1, usecols=range(1, J + 1), delimiter='\t', dtype=np.float64) for mat in
                       simmat_v])

        # Nearest neighbors truncation + regularization
        def truncate_kernel(K):
            idx = np.argsort(-K, axis=1)
            for i in range(K.shape[0]):
                K[i, idx[i, 5:]] = 0
            K += K.T
            K -= (np.real_if_close(np.min(np.linalg.eigvals(K)) - 0.1)) * np.eye(K.shape[0])

        for i in range(len(Ku)):
            truncate_kernel(Ku[i])

        for i in range(len(Kv)):
            truncate_kernel(Kv[i])

        # Load CV folds
        # folds = []
        # with open("DataOfVB-MK-LMF/nr/cv/nr_all_folds_cvs1.txt") as f:
        #     for i in f.readlines():
        #         rec = i.strip().split(",")
        #         ln = len(rec) // 2
        #         folds += [[(int(rec[j * 2]) - 1, int(rec[j * 2 + 1]) - 1) for j in range(ln)]]

        # Latent dims and augmented Bernoulli parameter
        L = 12
        c = 3.0

        # Insert your favorite neural network here
        def nn(Uw1, Vw1):
            return tf.matmul(Uw1, Vw1, transpose_a=True)

        # Augmented Bernoulli distribution
        #  sampling is not used and therefore omitted

        class dAugmentedBernoulli(Distribution):
            def __init__(self, logits, c, obs,
                         validate_args=False,
                         allow_nan_stats=True,
                         name="AugmentedBernoulli"):
                parameters = locals()
                with tf.name_scope(name):
                    with tf.control_dependencies([]):
                        self._logits = tf.identity(logits)
                        self._c = tf.identity(c)
                        self._obs = tf.identity(obs)
                super(dAugmentedBernoulli, self).__init__(dtype=tf.int32, validate_args=validate_args,
                                                          allow_nan_stats=allow_nan_stats,
                                                          reparameterization_type=tf.contrib.distributions.NOT_REPARAMETERIZED,
                                                          parameters=parameters,
                                                          graph_parents=[self._logits, self._c, self._obs], name=name)

            def _log_prob(self, event):
                event = tf.cast(event, tf.float32)
                cond = self._logits >= 0
                neg_abs = tf.where(cond, -self._logits, self._logits)
                sig = ((self._c - 1.0) * tf.cast(event, tf.float32) + 1.0) * tf.log1p(tf.exp(neg_abs))
                return self._obs * tf.where(cond, (event - 1) * self._logits - sig,
                                            self._c * event * self._logits - sig)

        # def __init__(self, *args, **kwargs):
        #     RandomVariable.__init__(self, *args, **kwargs)
        def ref(self, *args, **kwargs):
            RandomVariable.__init__(self, *args, **kwargs)

        AugmentedBernoulli = type("AugmentedBernoulli", (RandomVariable, dAugmentedBernoulli), {'__init__': ref})

        # Construct VB-MK-LMF model
        # Gamma distributions can lead to very noisy gradients so LogNormals are used instead

        def construct_model():
            nku = len(Ku)
            nkv = len(Kv)

            obs = tf.placeholder(tf.float32, R_.shape)

            Ug = TransformedDistribution(distribution=Normal(tf.zeros([nku]), tf.ones([nku])),
                                         bijector=tf.contrib.distributions.bijectors.Exp())
            Vg = TransformedDistribution(distribution=Normal(tf.zeros([nkv]), tf.ones([nkv])),
                                         bijector=tf.contrib.distributions.bijectors.Exp())

            Ua = TransformedDistribution(distribution=Normal(tf.zeros([1]), tf.ones([1])),
                                         bijector=tf.contrib.distributions.bijectors.Exp())
            Va = TransformedDistribution(distribution=Normal(tf.zeros([1]), tf.ones([1])),
                                         bijector=tf.contrib.distributions.bijectors.Exp())

            cKu = tf.cholesky(Ku + tf.eye(I) / Ua)  # TODO: rank 1 chol update
            cKv = tf.cholesky(Kv + tf.eye(J) / Va)

            Uw1 = MultivariateNormalTriL(tf.zeros([L, I]),
                                         tf.reduce_sum(cKu / tf.reshape(tf.sqrt(Ug), [nku, 1, 1]), axis=0))
            Vw1 = MultivariateNormalTriL(tf.zeros([L, J]),
                                         tf.reduce_sum(cKv / tf.reshape(tf.sqrt(Vg), [nkv, 1, 1]), axis=0))

            logits = nn(Uw1, Vw1)
            R = AugmentedBernoulli(logits=logits, c=c, obs=obs, value=tf.cast(logits > 0, tf.int32))

            qUg = TransformedDistribution(distribution=NormalWithSoftplusScale(tf.Variable(tf.zeros([nku])),
                                                                               tf.Variable(tf.ones([nku]))),
                                          bijector=tf.contrib.distributions.bijectors.Exp())
            qVg = TransformedDistribution(distribution=NormalWithSoftplusScale(tf.Variable(tf.zeros([nkv])),
                                                                               tf.Variable(tf.ones([nkv]))),
                                          bijector=tf.contrib.distributions.bijectors.Exp())
            qUa = TransformedDistribution(distribution=NormalWithSoftplusScale(tf.Variable(tf.zeros([1])),
                                                                               tf.Variable(tf.ones([1]))),
                                          bijector=tf.contrib.distributions.bijectors.Exp())
            qVa = TransformedDistribution(distribution=NormalWithSoftplusScale(tf.Variable(tf.zeros([1])),
                                                                               tf.Variable(tf.ones([1]))),
                                          bijector=tf.contrib.distributions.bijectors.Exp())
            qUw1 = MultivariateNormalTriL(tf.Variable(tf.zeros([L, I])), tf.Variable(tf.eye(I)))
            qVw1 = MultivariateNormalTriL(tf.Variable(tf.zeros([L, J])), tf.Variable(tf.eye(J)))

            return obs, Ug, Vg, Ua, Va, cKu, cKv, Uw1, Vw1, R, qUg, qVg, qUa, qVa, qUw1, qVw1

        # auroc_all = []
        # aupr_all = []
        # for f in folds:
            # Edward does not delete nodes so we have to reset the graph manually
        ed.get_session().close()
        tf.reset_default_graph()
        obs, Ug, Vg, Ua, Va, cKu, cKv, Uw1, Vw1, R, qUg, qVg, qUa, qVa, qUw1, qVw1 = construct_model()

        # Hide test examples
        # cv = np.zeros((I, J), dtype=np.bool)
        # for i in f:
        #     cv[i[1], i[0]] = True
        data = np.copy(R_)
        # data[cv] = 0

        # Construct observation matrix for the augmented Bernoulli distribution
        obs_ = (np.logical_and.outer(np.any(data > 0, axis=1), np.any(data > 0, axis=0)) * 1).astype(np.float32)

        # Variational approximation using BBVI
        inference = ed.KLqp({Uw1: qUw1, Vw1: qVw1, Ug: qUg, Vg: qVg, Ua: qUa, Va: qVa}, data={R: data, obs: obs_})
        inference.initialize(n_samples=10, n_iter=3000)
        tf.global_variables_initializer().run()
        for _ in range(inference.n_iter):
            info_dict = inference.update()
            inference.print_progress(info_dict)
        inference.finalize()

        # Evaluation
        res = tf.nn.sigmoid(nn(qUw1.mean(), qVw1.mean()) ** c).eval()
        res = np.transpose(res)
        self.predictR=res
        #     prc, rec, _ = precision_recall_curve(R_[cv], res[cv])
        #     fpr, tpr, _ = roc_curve(R_[cv], res[cv])
        #
        #     auroc = auc(fpr, tpr, reorder=True)
        #     aupr = auc(rec, prc, reorder=True)
        #     auroc_all += [auroc]
        #     aupr_all += [aupr]
        #     print("{}\t{}\t{}".format(aupr, auroc, f))
        # print("Overall\nAUPR: {}, AUROC: {}".format(np.mean(aupr_all), np.mean(auroc_all)))

    def predict_scores(self, test_data, N):
        1-1

    def evaluation(self, test_data, test_label):

        score = self.predictR
        import pandas as pd
        score = pd.DataFrame(score)
        score.to_csv('../data/datasets/EnsambleDTI/vbmklmf_'+str(self.dataset)+'_s'+
                      str(self.cvs)+'_'+str(self.seed)+'_'+str(self.num)+'.csv', index=False)
        score = self.predictR[test_data[:, 0], test_data[:, 1]]
        prec, rec, thr = precision_recall_curve(test_label,np.array(score))
        aupr_val = auc(rec, prec)
       
        fpr, tpr, thr = roc_curve( test_label,np.array(score))
        auc_val = auc(fpr, tpr)
        print("AUPR: " + str(aupr_val) + ", AUC: " + str(auc_val))
        return aupr_val, auc_val
    def __str__(self):
        return "Model: VB-MK-LMF"
