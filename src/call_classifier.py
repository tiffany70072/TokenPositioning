#import matplotlib.pyplot as plt
import numpy as np
#import os
#import pdb
#import pickle
#import sys

#from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV, RFE
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

#import analysis

def call_rfe(x, y, n_feature=None):
	svc = SVC(kernel="linear", C=1)
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
	if n_feature is None:
		rfe = RFE(svc, step=1)
	else:
		rfe = RFE(svc, n_features_to_select=n_feature, step=1)
	rfe.fit(x_train, y_train)
	pred = rfe.predict(x_test)
	print ("\t(RFE Acc) Train = %.2f" % (np.sum(rfe.predict(x_train) == y_train) / y_train.shape[0]), end=', ')
	print ("\tTest = %.2f" % (np.sum(pred == y_test) / y_test.shape[0]), end=", ")
	print ("\tFeatures:", np.where(rfe.ranking_ == 1)[0])
	return np.where(rfe.ranking_ == 1)[0]


def call_recursive_rfe(x, y, threshold=0.7, one_threshold=0.0, N=8, max_count=80):
	import warnings
	warnings.filterwarnings('ignore', 'Solver terminated early.*')

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
	feature_list = []
	count = 0
	while count < max_count:
		count += N
		print("\tcount =", count, end=", ")
		svc = SVC(kernel="linear", C=1, max_iter=300)
		rfe = RFE(svc, n_features_to_select=N, step=1)
		rfe.fit(x_train, y_train)
		print ("\t(RFE, one) Train accuracy = %.3f" % (np.sum(rfe.predict(x_train) == y_train) / y_train.shape[0]), end=', ')
		test_acc = np.sum(rfe.predict(x_test) == y_test) / y_test.shape[0]
		print ("\tTest accuracy = %.3f" % test_acc)
		
		if test_acc < one_threshold:
			break
		selected_feature = list(np.where(rfe.ranking_ == 1)[0])
		x_train[:, selected_feature] *= 0
		x_test[:, selected_feature] *= 0
		feature_list += selected_feature

	return feature_list
