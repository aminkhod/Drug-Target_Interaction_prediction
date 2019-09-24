'''
[1] Yong Liu, Min Wu, Chunyan Miao, Peilin Zhao, Xiao-Li Li, "Neighborhood Regularized Logistic Matrix Factorization for Drug-target Interaction Prediction", under review.
'''

import pandas as pd
from keras.layers import Input, Dense
from keras.models import Model
from keras import optimizers
from sklearn.metrics import *
import numpy as np
import tensorflow as tf
# from DTHybrid import DTHYBRID
# from netcbp.netcbp import NetCBP
# from blm import BLMNII
# from brdti import BRDTI
# from cmf import CMF
# from daspfind import DASPFIND
# from ddr import DDR
# from dnilmf import DNILMF
# from functions import *
# from kbmf2k.kbmf import KBMF
# from kronrlsmkl import KronRLsMKL
# from netlaprls import NetLapRLS
# from new_pairs import novel_prediction_analysis
# from nrlmf import NRLMF
# from vbmklmf import VBMKLMF
# from wnngip import WNNGIP
# from pudt.pudt import PUDT
# from GRMF.GRMF import GRMF
from functions import *

class EnsambleDTI:

	def __init__(self, args, dataset):
		self.dataset = dataset


	def fix_model(self, W, intMat, drugMat, targetMat, test_data, test_label,num, cv, dataset, seed=None):
		# cvs, sp_arg, model_settings, predict_num = cv, 1, [], 0
		self.intMat=intMat
		x, y = np.where(self.intMat > 0)
		self.train_drugs, self.train_targets = set(x.tolist()), set(y.tolist())
		self.dsMat = drugMat
		self.tsMat = targetMat
		# score = np.zeros((np.size(self.intMat[:,1]), np.size(self.intMat[1,:])))
		methods=[ 'blmniis',  'daspfind','dthybrid', 'grmf', 'netcbp', 'kbmf', 'dnilmf',
				  'kronrlsmkl', 'netlaprls', 'wnngip', 'vbmklmf']
		# 'cmf', 'nrlmf',
		# methods = ['grmf']
		# methods = ['netcbp']
		#methods = ['kbmf']
		#methods = ['dnilmf']
		# methods = ['ddr']
		# methods = ['brdti']
		# methods = ['vbmklmf']
		# # methods = ['pudt']
		# for method in methods:
		# 	if method == 'nrlmf':
		# 		args = {'c': 5, 'K1': 5, 'K2': 5, 'r': 50, 'lambda_d': 0.125, 'lambda_t': 0.125,
		# 	            'alpha': 0.25, 'beta': 0.125, 'theta': 0.5, 'max_iter': 100}
		# 	if method == 'daspfind':
		# 		args = {'alpha': 2.26}
		# 	if method == 'pudt':
		# 		args = {""}
		# 	if method == 'grmf':
		# 		args = {""}
		# 	if method == 'netcbp':
		# 		args = {""}
		# 	if (method == 'dnilmf'):
		# 		args = {""}
		# 	if (method == 'dthybrid'):
		# 		args = {""}
		# 	if (method == 'kronrlsmkl'):
		# 		args = {""}
		# 	if (method == 'ddr'):
		# 		args = {""}
		# 	if (method == 'vbmklmf'):
		# 		args = {""}
		#
		# 	if (method == 'brdti') | (method == 'inv_brdti'):
		# 		args = {
		# 			'D': 100,
		# 			'learning_rate': 0.1,
		# 			'max_iters': 100,
		# 			'simple_predict': False,
		# 			'bias_regularization': 1,
		# 			'global_regularization': 10 ** (-2),
		# 			"cbSim": "knn",
		# 			'cb_alignment_regularization_user': 1,
		# 			'cb_alignment_regularization_item': 1}
		#
		# 	if method == 'netlaprls':
		# 		args = {'gamma_d': 10, 'gamma_t': 10, 'beta_d': 1e-5, 'beta_t': 1e-5}
		# 	if method == 'blmnii':
		# 		args = {'alpha': 0.7, 'gamma': 1.0, 'sigma': 1.0, 'avg': False}
		# 	if method == 'wnngip':
		# 		args = {'T': 0.8, 'sigma': 1.0, 'alpha': 0.8}
		# 	if method == 'kbmf':
		# 		args = {'R': 50}
		# 	if method == 'cmf':
		# 		args = {'K': 50, 'lambda_l': 0.5, 'lambda_d': 0.125, 'lambda_t': 0.125, 'max_iter': 30}
		# 	tic = time.clock()
		# 	if (method == "netcbp"):
		# 		model = NetCBP()
		# 	# if (method=="ndaf"):
		# 	#     model = NDAF()
		# 	if (method == "grmf"):
		# 		model = GRMF(cv=cvs)
		# 	if (method == "pudt"):
		# 		model = PUDT(dataset=self.dataset)
		# 	if (method == "vbmklmf"):
		# 		model = VBMKLMF(dataset=self.dataset, cvs=cvs)
		# 	if (method == 'dnilmf'):
		# 		model = DNILMF(dataset=self.dataset)
		# 	if (method == 'kronrlsmkl'):
		# 		model = KronRLsMKL(dataset=self.dataset)
		# 	if (method == 'dthybrid'):
		# 		model = DTHYBRID(dataset=self.dataset)
		# 	if (method == 'daspfind'):
		# 		model = DASPFIND(alpha=args['alpha'])
		# 	if (method == 'brdti') | (method == 'inv_brdti'):
		# 		# model = BRDTI(D=args['D'],learning_rate= args['learning_rate'],max_iters=args['max_iters'],simple_predict=args['simple_predict'],bias_regularization=args['bias_regularization'],global_regularization=args['global_regularization'],cbSim=args['cbSim'],cb_alignment_regularization_user=args['cb_alignment_regularization_user'],cb_alignment_regularization_item=args['cb_alignment_regularization_item'])
		# 		model = BRDTI(args)
		# 	if method == 'nrlmf':
		# 		model = NRLMF(cfix=args['c'], K1=args['K1'], K2=args['K2'], num_factors=args['r'],
		# 					  lambda_d=args['lambda_d'], lambda_t=args['lambda_t'], alpha=args['alpha'], beta=args['beta'],
		# 					  theta=args['theta'], max_iter=args['max_iter'])
		# 	if method == 'ddr':
		# 		model = DDR(dataset=self.dataset, cv=cvs)
		# 	if method == 'netlaprls':
		# 		model = NetLapRLS(gamma_d=args['gamma_d'], gamma_t=args['gamma_t'], beta_d=args['beta_t'],
		# 						  beta_t=args['beta_t'])
		# 	if method == 'blmnii':
		# 		model = BLMNII(alpha=args['alpha'], gamma=args['gamma'], sigma=args['sigma'], avg=args['avg'])
		# 	if method == 'wnngip':
		# 		model = WNNGIP(T=args['T'], sigma=args['sigma'], alpha=args['alpha'])
		# 	if method == 'kbmf':
		# 		model = KBMF(num_factors=args['R'])
		# 	if method == 'cmf':
		# 		model = CMF(K=args['K'], lambda_l=args['lambda_l'], lambda_d=args['lambda_d'], lambda_t=args['lambda_t'],
		# 					max_iter=args['max_iter'])
		# 	if (method == 'ensambledti'):
		# 		model = EnsambleDTI(args=args, dataset=self.dataset)
		# 	cmd = str(model)
		# 	print("Dataset:" + self.dataset + " CVS:" + str(cvs) + "\n" + cmd)
		# 	model.fix_model(W, intMat, drugMat, targetMat,num, cv, dataset, seed)
		# 	aupr_val, auc_val = model.evaluation(test_data, test_label)
		ensambleFeature = []
		ensambleFeature =pd.DataFrame(ensambleFeature)
		buf = []
		for method in methods:
			buf = pd.read_csv('../data/datasets/EnsambleDTI/'+str(method)+ '_'+dataset+
							  '_s'+str(cv)+'_'+str(seed)+'_'+str(num)+'.csv')
			ensambleFeature[str(method)] = buf.values.flatten()
		# print(ensambleFeature)
		x,y =ensambleFeature.shape
		self.r, self.c = buf.shape
		#######Autoencoder############

		# this is our input placeholder
		input_img = Input(shape=(y,))
		# "encoded" is the encoded representation of the input
		encoded = Dense(y, activation='relu')(input_img)
		encoded = Dense(7, activation='relu')(encoded)
		encoded = Dense(4, activation='relu')(encoded)
		encoded = Dense(1, activation='relu')(encoded)

		decoded = Dense(4, activation='relu')(encoded)
		decoded = Dense(7, activation='relu')(decoded)
		decoded = Dense(y, activation='relu')(decoded)

		# this model maps an input to its reconstruction
		autoencoder = Model(input_img, decoded)

		autoencoder.compile(optimizer=optimizers.Adadelta(lr=0.002), loss='mse')
		# optimizer=Adadelta(lr=0.004)
		# Adam(lr=0.02)

		autoencoder.fit(x=ensambleFeature, y=ensambleFeature,
						epochs=400, batch_size=int(x/4),
						validation_split=0.1)
		weight1 = autoencoder.layers[1].get_weights()
		weight2 = autoencoder.layers[2].get_weights()
		weight3 = autoencoder.layers[3].get_weights()
		weight4 = autoencoder.layers[4].get_weights()
		# weight0 = autoencoder.layers[0].get_weights()
		# weight0 = autoencoder.layers[0].get_weights()


		i1 = tf.matmul( tf.cast(ensambleFeature, tf.float32),weight1[0])+weight1[1]
		i2 = tf.matmul(i1,weight2[0])+weight2[1]
		i3 = tf.matmul(i2, weight3[0])+weight3[1]
		i4 = tf.matmul(i3, weight4[0])+weight4[1]



		# j1 = tf.matmul( tf.cast(ensambleFeature, tf.float32),weight1[0])+weight1[1]
		# j2 = tf.matmul(j1,weight2[0])+weight2[1]
		# j3 = tf.matmul(j2, weight3[0])+weight3[1]

		with tf.Session() as ses:
			encoded_train = ses.run(i4)
		# with tf.Session() as ses1:
		# 	encoded_test = ses1.run(j3)
		# "encoded" is the encoded representation of the input


		# encoded = Dense(6, activation='relu')(input_img)
		# encoded = Dense(1, activation='relu')(encoded)
		#
		# decoded = Dense(6, activation='relu')(encoded)
		# decoded = Dense(y, activation='sigmoid')(decoded)
		#
		# # this model maps an input to its reconstruction
		# autoencoder = Model(input_img, decoded)
		# # this model maps an input to its encoded representation
		# encoder = Model(input_img, encoded)
		# #mean_squared_error
		# autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
		#
		# autoencoder.fit(ensambleFeature, ensambleFeature,
		# 				epochs=x*2,
		# 				batch_size=int(x/10))

		self.predictR = encoded_train



	def predict_scores(self, test_data, N):
		return 1 + 1


	def evaluation(self, test_data, test_label):
		self.predictR = np.reshape(self.predictR,(self.r, self.c))
		score = self.predictR[test_data[:, 0], test_data[:, 1]]
		prec, rec, thr = precision_recall_curve(test_label, np.array(score))
		aupr_val = auc(rec, prec)
		fpr, tpr, thr = roc_curve(test_label, np.array(score))
		auc_val = auc(fpr, tpr)
		print("AUPR: " + str(aupr_val) + ", AUC: " + str(auc_val))

		return aupr_val, auc_val


	def __str__(self):
			return "Model: EnsambleDTI"

# from sklearn.linear_model import LogisticRegression
# logreg = LogisticRegression(solver='lbfgs')
# LogregModel = logreg.fit(encoded_train, y_train)
# predicts=LogregModel.predict(encoded_test)
#
# fpr, tpr, thresholds = roc_curve(y_test,predicts)
# log_encodauc= auc(fpr, tpr)
# print("AUC with Autoencoder:",log_encodauc)
