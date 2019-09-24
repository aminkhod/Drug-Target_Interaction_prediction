from sklearn.metrics import precision_recall_curve,auc
from sklearn.preprocessing import MaxAbsScaler
import numpy as np
from sklearn.ensemble import RandomForestClassifier
#from  sklearn import linear_model
#import itertools




def run_classification_configuration(data,labels,test_idx,trees,cw,c,performance=True):
	All_scores = []
	# data= np.transpose(data)
	length = len(data[0])
	#print len(data)
	total_AUPR_training = 0
	total_AUPR_testing = 0
	labels = np.array(labels)
	folds_AUPR = []
	for fold_data,test_idx_fold in zip(data,test_idx):
		train_idx_fold = []
		for idx in range(length):
			if idx not in test_idx_fold:
				train_idx_fold.append(idx)



		fold_data = np.array(fold_data)
		test_idx_fold = np.array(test_idx_fold)
		train_idx_fold = np.array(train_idx_fold)


	X_train, X_test = fold_data[train_idx_fold,], fold_data[test_idx_fold,]
	y_train, y_test = labels[train_idx_fold], labels[test_idx_fold]


	max_abs_scaler = MaxAbsScaler()
	X_train_maxabs_fit = max_abs_scaler.fit(X_train)

	X_train_maxabs_transform = max_abs_scaler.transform(X_train)

	X_test_maxabs_transform = max_abs_scaler.transform(X_test)

	#counts = collections.Counter(y_train)

	#ratio = 1.0*counts[0]/counts[1]
	#print ratio


	#rf  = xgboost.XGBClassifier(n_estimators=trees,scale_pos_weight=ratio,learning_rate=cw,max_depth=c)
	rf = RandomForestClassifier(n_estimators=trees ,n_jobs=6,criterion = c,class_weight="balanced",random_state=1357)

	rf.fit(X_train_maxabs_transform, y_train)
	try:
		scores_training = rf.decision_function(X_train_maxabs_transform)
		scores_testing =  rf.decision_function(X_test_maxabs_transform)
	except:
		scores_training = rf.predict_proba(X_train_maxabs_transform)[:, 1]
		scores_testing =  rf.predict_proba(X_test_maxabs_transform)[:, 1]


	y_pred = rf.predict(X_test_maxabs_transform)




	if performance:
		precision_training, recall_training, _ = precision_recall_curve(y_train, scores_training, pos_label=1)
		precision_testing, recall_testing, _ =   precision_recall_curve(y_test, scores_testing, pos_label=1)
		AUPR_training = auc(recall_training,precision_training)
		AUPR_testing = auc(recall_testing, precision_testing)
		folds_AUPR.append(AUPR_testing)


		#print AUPR_testing
		total_AUPR_training+=AUPR_training
		total_AUPR_testing += AUPR_testing

	else:
		All_scores.append(scores_testing)
	if performance:
		Avg_AUPR_training = 1.0*total_AUPR_training
		Avg_AUPR = 1.0*total_AUPR_testing/len(data)

	if performance:
		return [Avg_AUPR_training,Avg_AUPR,folds_AUPR,c]
	else:
		# return All_scores
		return scores_testing , scores_training




def run_classification(data,labels,test_idx):
	learning_rate = [0.01]#[0.01,0.02,0.03,0.04,0.05,0.05,0.07,0.08,0.09,0.01]
	max_depth = [3,4,5,6]

	#no_trees = [5,10,15]#range(5,62,2)
	no_trees = [100,200,300,400]
	criterion = ["gini","entropy"]
	result = []
	for cw in learning_rate:
		for c in criterion:
			for t in no_trees:
				parameter_result = run_classification_configuration(data,labels,test_idx,t,cw,c)
				#print 'no_trees',t, 'max_depth',c, 'learning_rate',cw
				#print 'total_AUPR_training:', round(AUPR_train,2)
				#print 'total_AUPR_testing:', round(AUPR_test, 2)
				result.append(parameter_result)
	result.sort(key=lambda x:x[0],reverse=True)
	return result[0]

