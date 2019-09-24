"""
Bayesian Ranking Prediction of Drug-Target Interactions - BRDTI
Implementation is based on largely extended Bayesian Personalized Ranking Matrix Factorization (BPR), obtained from github.com/gamboviol/bpr
[1] Rendle, S. et al. (2009) BPR: Bayesian personalized ranking from implicit feed-back. In UAI 2009. AUAI Press, 452-461
[2] Peska, L. Buza, K. (2016) Drug-Target Interaction Prediction: a Bayesian Ranking Approach. Submitted to Bioinformatics, http://www.ksi.mff.cuni.cz/~peska/BRDTI/author_version.pdf

Parameters:
    D = number of latent factors
    learning_rate = initial learning rate (further tuned via bold driver heuristics)
    max_iters = total iterations executed by the stochastic gradient descend alg.
    global_regularization = lambda_g from [2]
    cb_alignment_regularization = lambda_c from [2]
    bias_regularization = regularization of target bias, held constant at [2]

    simple_predict = boolean whether to use content smoothing approach from [2]
    cbSim = nominal param, if/which changes should be made with the similarity matrices (e.g. knn reduction)

"""
from __future__ import division
import numpy as np
import scipy.sparse as sp
from math import exp    
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc
from functions import normalized_discounted_cummulative_gain

class BRDTI:

    def __init__(self,args):

        self.D = args["D"]
        self.orig_learning_rate = args["learning_rate"]
        self.learning_rate = self.orig_learning_rate
        
        self.max_iters = args["max_iters"]
        self.user_regularization = args["global_regularization"]        
        self.bias_regularization = args["global_regularization"]* args["bias_regularization"]
        self.user_cb_alignment_regularization = self.user_regularization * args["cb_alignment_regularization_user"]    
        
        self.simple_predict = args["simple_predict"]
        self.cbSim = args["cbSim"]       
        
                
        self.positive_item_regularization = self.user_regularization
        self.negative_item_regularization = self.user_regularization             
        self.item_cb_alignment_regularization = self.user_cb_alignment_regularization   
    
    def get_nearest_neighbors(self, S, size=5):
        m, n = S.shape
        X = np.zeros((m, n))
        for i in range(m):
            ii = np.argsort(np.asarray(S[i, :]).reshape(-1))[::-1][:min(size, n)]
            ii = ii[0:size]           
            X[i, ii] = S[i, ii]
        return X   
    
 
    def fix_model(self, W, intMat, drugMat, targetMat,num, cvs, dataset, seed=None):
        self.dataset = dataset
        self.num = num
        self.cvs = cvs
        self.seed = seed
        self.learning_rate = self.orig_learning_rate
        self.num_drugs, self.num_targets = intMat.shape
        dt = W*intMat
        self.intMat = dt
        data = sp.csr_matrix(dt)
        self.data = data
        x, y = np.where(dt > 0)
        self.train_drugs, self.train_targets = set(x.tolist()), set(y.tolist())
        self.dinx = np.array(list(self.train_drugs))
        self.tinx = np.array(list(self.train_targets)) 
        
        np.random.seed(seed)
            
        #similarity regularization
        uSim = np.matrix(drugMat)
        iSim = np.matrix(targetMat)
        
        #we do not want to consider similarity of items with themselves thus = 0 
        uSim -= np.eye(uSim.shape[0])      
        iSim -= np.eye(iSim.shape[0])
        
        self.kd = np.zeros(uSim.shape)
        self.kt = np.zeros(iSim.shape)
        self.kd[:,self.dinx] = 1
        self.kt[:,self.tinx] = 1        
        
        #if we do not want to consider uknown drugs/tragets similarity in evaluation, uncomment this          
        """ 
        uSim = np.asmatrix(np.asarray(uSim)*self.kd)
        iSim = np.asmatrix(np.asarray(iSim)*self.kt) 
        
        """
                      
        if self.cbSim == "knn":
            uSim = self.get_nearest_neighbors(uSim)
            iSim = self.get_nearest_neighbors(iSim)
            uSim = np.asmatrix(uSim) 
            iSim = np.asmatrix(iSim)
        
        uSum = (uSim.sum() / uSim.shape[0])   
        uSim = (1/uSum) * uSim
        iSum = (iSim.sum() / iSim.shape[0])   
        iSim = (1/iSum) * iSim   

  
        #remove unsimilar items from the matrix
        #this did not prove to be succesful
        """
        uP50 = np.percentile(uSim, 50)
        uP75 = np.percentile(uSim, 75)
        uP90 = np.percentile(uSim, 90)
        
        iP50 = np.percentile(iSim, 50)
        iP75 = np.percentile(iSim, 75)
        iP90 = np.percentile(iSim, 90)
        
        #print(uP50, uP75, uP90)
        #print(iP50, iP75, iP90)
        
        uSim[uSim < uP50] = 0
        iSim[iSim < iP50] = 0
        """        
        
        self.train(data, uSim, iSim, seed)
  
    def train(self,data, uSim, iSim, seed):
        """train model
        data: user-item matrix as a scipy sparse matrix
              users and items are zero-indexed
        userSim: matrix of user similarities
        itemSim: matrix of item similarities
        """
        self.init(data, uSim, iSim, seed)      
        
        act_loss = self.loss()
        n_samples = self.data.nnz
        
        for it in range(self.max_iters):
            users, pos_items, neg_items = self._uniform_user_sampling( n_samples)
            for u,i,j in zip(users, pos_items, neg_items):
                self.update_factors(u,i,j)

            #execute bold driver learning  after each epoch  
            new_loss =  self.loss()
            if new_loss < act_loss:
                self.learning_rate = self.learning_rate * 1.1
            else:
                self.learning_rate = self.learning_rate * 0.5
            act_loss = new_loss
        
    def init(self,data, uSim, iSim, seed):
        self.data = data
        self.uSim = uSim
        self.iSim = iSim
        
        self.num_users,self.num_items = self.data.shape
        self.user_bias = np.zeros(self.num_users)
        self.item_bias = np.zeros(self.num_items)
        
        if seed is None:
            self.user_factors = np.sqrt(1/float(self.D)) * np.random.normal(size=(self.num_users,self.D))
            self.item_factors = np.sqrt(1/float(self.D)) * np.random.normal(size=(self.num_items,self.D))
        else:
            prng = np.random.RandomState(seed)
            self.user_factors = np.sqrt(1/float(self.D)) * prng.normal(size=(self.num_users,self.D))
            self.item_factors = np.sqrt(1/float(self.D)) * prng.normal(size=(self.num_items,self.D))
               
        self.create_loss_samples(data)

    def create_loss_samples(self, data):
        
        num_loss_samples = int(100*self.num_users**0.5)
        users, pos_items, neg_items = self._uniform_user_sampling( num_loss_samples)
        self.loss_samples = list(zip(users, pos_items, neg_items))

    def update_factors(self,u,i,j,update_u=True,update_i=True):
        """apply SGD update"""
        update_j = True

        x = self.item_bias[i] - self.item_bias[j] \
            + np.dot(self.user_factors[u,:],self.item_factors[i,:]-self.item_factors[j,:])


        if x > 200:
            z = 0
        if x < -200:
            z = 1
        else:
            ex = exp(-x)
            z = ex/(1.0 + ex)


        # update bias terms
        if update_i:
            d = z - self.bias_regularization * self.item_bias[i]
            self.item_bias[i] = self.item_bias[i]+ self.learning_rate * d
        if update_j:
            d = -z - self.bias_regularization * self.item_bias[j]
            self.item_bias[j] = self.item_bias[j]+ self.learning_rate * d

        if update_u:
            d = (self.item_factors[i,:]-self.item_factors[j,:])*z - self.user_regularization*self.user_factors[u,:]
            if self.user_cb_alignment_regularization > 0:
                #code for updating content alingment - based on similarity matrix
                alignmentVectorU = np.dot(self.uSim[u,:], self.user_factors)
                alignmentSumU = np.sum(self.uSim[u,:])

                d = d + 2*self.user_cb_alignment_regularization * (alignmentVectorU - (alignmentSumU * self.user_factors[u,:]) )

            self.user_factors[u,:] += self.learning_rate * np.asarray(d).reshape(-1)

        if update_i:
            d = self.user_factors[u,:]*z - self.positive_item_regularization*self.item_factors[i,:]
            if self.item_cb_alignment_regularization > 0:
                #code for updating content alingment - based on similarity matrix
                alignmentVectorI = np.dot(self.iSim[i,:], self.item_factors)
                alignmentSumI = np.sum(self.iSim[i,:])

                d = d + 2*self.item_cb_alignment_regularization * (alignmentVectorI - (alignmentSumI * self.item_factors[i,:]))

            self.item_factors[i,:] += self.learning_rate * np.asarray(d).reshape(-1)

        if update_j:
            d = -self.user_factors[u,:]*z - self.negative_item_regularization*self.item_factors[j,:]
            if self.user_cb_alignment_regularization > 0:
                #code for updating content alingment - based on similarity matrix
                alignmentVectorJ = np.dot(self.iSim[j,:], self.item_factors)
                alignmentSumJ = np.sum(self.iSim[j,:])

                d = d + 2*self.item_cb_alignment_regularization * (alignmentVectorJ - (alignmentSumJ * self.item_factors[j,:]))

            self.item_factors[j,:] += self.learning_rate * np.asarray(d).reshape(-1)

    def _uniform_user_sampling(self, n_samples):
        """
          Creates `n_samples` random samples from training data for performing Stochastic
          Gradient Descent. We start by uniformly sampling users,
          and then sample a positive and a negative item for each
          user sample.
        """

        sgd_users = np.random.choice(list(self.train_drugs),size=n_samples)
        sgd_pos_items, sgd_neg_items = [], []
        for sgd_user in sgd_users:
            pos_item = np.random.choice(self.data[sgd_user].indices)
            neg_item = np.random.choice(list(self.train_targets - set(self.data[sgd_user].indices)))
            sgd_pos_items.append(pos_item)
            sgd_neg_items.append(neg_item)

        return sgd_users, sgd_pos_items, sgd_neg_items

    def loss(self):
        ranking_loss = 0;
        for u,i,j in self.loss_samples:
            x = self.predict(u,j) - self.predict(u,i)

            if x > 200:
                rl = 0
            if x < -200:
                rl = 1
            else:
                ex = exp(-x)
                rl = 1.0/(1.0+ex)
            ranking_loss += rl

        complexity = self.complexity()

        return ranking_loss + complexity


    def complexity(self):
        complexity = 0
        for u,i,j in self.loss_samples:
            complexity += self.user_regularization * np.dot(self.user_factors[u],self.user_factors[u])
            complexity += self.positive_item_regularization * np.dot(self.item_factors[i],self.item_factors[i])
            complexity += self.negative_item_regularization * np.dot(self.item_factors[j],self.item_factors[j])

            complexity += -(self.item_cb_alignment_regularization * np.dot(np.dot(self.item_factors,self.item_factors[i,:]), self.iSim[:,i])[0,0] )
            complexity += -(self.item_cb_alignment_regularization * np.dot(np.dot(self.item_factors,self.item_factors[j,:]), self.iSim[:,j])[0,0] )
            complexity += -(self.user_cb_alignment_regularization * np.dot(np.dot(self.user_factors,self.user_factors[u,:]), self.uSim[:,u])[0,0] )


            complexity += self.bias_regularization * self.item_bias[i]**2
            complexity += self.bias_regularization * self.item_bias[j]**2

        return complexity

    def predict(self,u,i):

        #predict only from learned factors, ignore neighbouring users and items even for novel ones
        if self.simple_predict:
            return self.item_bias[i] + self.user_bias[u] + np.dot(self.user_factors[u],self.item_factors[i])

        elif (u not in self.train_drugs) & (i in self.train_targets):
            alignmentMatrixU = np.dot(self.uSim[u,:], self.user_factors)
            alignmentVectorU = alignmentMatrixU/np.sum(self.uSim[u,:])
            return self.item_bias[i] + np.mean(self.user_bias) + np.sum(np.array(alignmentVectorU[:]).flatten() * np.array(self.item_factors[i,:]).flatten())

        elif (i not in self.train_targets) & (u in self.train_drugs):
            alignmentMatrixI = np.dot(self.iSim[i,:], self.item_factors)
            alignmentVectorI = alignmentMatrixI/np.sum(self.iSim[i,:])
            return np.mean(self.item_bias) + self.user_bias[u] +  np.sum(np.array(alignmentVectorI[:]).flatten() * np.array(self.user_factors[u,:]).flatten())

        elif (i not in self.train_targets) & (u not in self.train_drugs):
            alignmentMatrixI = np.dot(self.iSim[i,:], self.item_factors)
            alignmentVectorI = alignmentMatrixI/np.sum(self.iSim[i,:])
            alignmentMatrixU = np.dot(self.uSim[u,:], self.user_factors)
            alignmentVectorU = alignmentMatrixU/np.sum(self.uSim[u,:])

            return np.mean(self.item_bias) +  np.mean(self.user_bias) + np.sum(np.array(alignmentVectorU[:]).flatten() * np.array(alignmentVectorI[:]).flatten())

        else:
            return self.item_bias[i] + self.user_bias[u] + np.dot(self.user_factors[u],self.item_factors[i])


    def predict_scores(self, test_data, N):
        scores = []
        for d, t in test_data:
            score = self.predict(d,t)
            if score > 200:
                scores.append(1)
            elif score < -200:
                scores.append(0)
            else:
                sc = np.exp(score)
                scores.append(sc/(1+sc))

        return np.array(scores)


    def evaluation(self, test_data, test_label):
        scores = []

        if self.D > 0:
            for d, t in self.intMat:
                score = self.predict(d,t)
                if score > 200:
                    scores.append(1)
                elif score < -200:
                    scores.append(0)
                else:
                    sc = np.exp(score)
                    scores.append(sc/(1+sc))


        import pandas as pd
        scores = pd.DataFrame(scores)
        scores.to_csv('../data/datasets/EnsambleDTI/brdti_'+str(self.dataset)+'_s'+
                      str(self.cvs)+'_'+str(self.seed)+'_'+str(self.num)+'.csv', index=False)
        # x, y = test_data[:, 0], test_data[:, 1]
        # test_data_T = np.column_stack((y,x))
        
        #ndcg = normalized_discounted_cummulative_gain(test_data, test_label, np.array(scores))
        #ndcg_inv = normalized_discounted_cummulative_gain(test_data_T, test_label, np.array(scores))
        if self.D > 0:
            for d, t in test_data:
                score = self.predict(d, t)
                if score > 200:
                    scores.append(1)
                elif score < -200:
                    scores.append(0)
                else:
                    sc = np.exp(score)
                    scores.append(sc / (1 + sc))
        prec, rec, thr = precision_recall_curve(test_label, scores)
        aupr_val = auc(rec, prec)
        fpr, tpr, thr = roc_curve(test_label, scores)
        auc_val = auc(fpr, tpr)
        print(aupr_val, auc_val)
        return aupr_val, auc_val#, ndcg, ndcg_inv
    
    

        
    
    def __str__(self):
        return "Model: BRDTI, factors:%s, learningRate:%s,  max_iters:%s, lambda_bias:%s, lambda_g:%s, lambda_c:%s, cbSim:%s, simple_predict:%s" % (self.D, self.learning_rate, self.max_iters, self.bias_regularization, self.user_regularization,  self.user_cb_alignment_regularization, self.cbSim, self.simple_predict)
    



