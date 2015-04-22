
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model, cross_validation, grid_search
from sklearn.metrics import roc_auc_score, accuracy_score
from matplotlib import pyplot as plt

data = pd.read_csv('strum_etas.csv', nrows = 395)
#data = data.reindex(np.random.permutation(data.index))
# data.columns
# Out[3]: Index([u'Subject', u'Label', u'EEG-Stim', u'EEG-Stim.1', u'EEG-Stim.2', u'EEG-Stim.3', u'EEG-Stim.4', u'EEG-Stim.5',
#  u'EEG-Stim.6', u'EEG-Stim.7', u'EEG-Stim.8', u'EEG-Stim.9', u'EEG-Stim.10', u'EEG-Stim.11', u'EEG-Stim.12', u'EEG-Stim.13', 
#  u'EEG-Stim.14', u'EEG-Stim.15', u'EEG-Stim.16', u'EEG-Stim.17', u'EEG-Stim.18', u'EEG-Stim.19', u'EEG-Stim.20', u'EEG-Stim.21', 
#  u'EEG-Stim.22', u'EEG-Stim.23', u'EEG-Stim.24', u'EEG-Stim.25', u'EEG-Stim.26', u'EEG-Stim.27', u'EEG-Stim.28', u'EEG-Cue',
#  u'EEG-Cue.1', u'EEG-Cue.2', u'EEG-Cue.3', u'EEG-Cue.4', u'EEG-Cue.5', u'EEG-Cue.6', u'EEG-Cue.7', u'EEG-Cue.8', u'EEG-Cue.9',
#  u'EEG-Cue.10', u'EEG-Cue.11', u'EEG-Cue.12', u'EEG-Cue.13', u'EEG-Cue.14', u'EEG-Cue.15', u'EEG-Cue.16', u'EEG-Cue.17', 
#  u'EEG-Cue.18', u'EEG-Cue.19', u'EEG-Cue.20', u'EEG-Cue.21', u'EEG-Cue.22', u'EEG-Cue.23', u'EEG-Cue.24', u'EEG-Cue.25',
#  u'EEG-Cue.26', u'EEG-Cue.27', u'EEG-Cue.28', u'Pupil', u'Pupil.1', u'Pupil.2', u'Pupil.3', u'Pupil.4', u'Pupil.5', 
#  u'Pupil.6', u'HR', u'RT'], dtype='object')

subjects_counts = data.Subject.value_counts()
subjects = subjects_counts.keys()

class strum_second_layer:

	def __init__(self, my_subject, modalities):
		global data
		self.subject = my_subject 
		data_single =  data[data.Subject == self.subject ]

		possible_modalities = ['EEG-Stim','EEG-Cue','Pupil','HR','RT']
		modality_index = []
		for modality in modalities:
			if modality in possible_modalities:
				modality_index.extend(filter((lambda x: x.startswith(modality)), data.columns))    

		self.y = data_single['Label'].values
		self.X = data_single[modality_index].values

	def fit(self):

		n_iter = 100 # the number of iterations should be more than that ... 
		c_range = np.logspace(-4, 4, 200)
		
		self.train_scores = np.zeros((10,len(c_range), n_iter))
		self.test_scores = np.zeros((10,len(c_range), n_iter))
		self.best_c_vec = np.zeros((10))

		cv1 = cross_validation.KFold(len(self.X), n_folds=10, indices=True, shuffle=True, random_state=4)
		self.predictions = []
		for idx_cv_1, (train1, test1) in enumerate(cv1):
			print 'Test set: ', idx_cv_1
			X_test = self.X[test1]
			y_test = self.X[test1]
			X_train = self.X[train1]
			y_train = self.y[train1]

			cv2 = cross_validation.ShuffleSplit(len(X_train), n_iter=n_iter, test_size=0.3)
			for idx_c, c in enumerate(c_range):
				for idx_cv_2, (train2, test2) in enumerate(cv2):
					logist = linear_model.LogisticRegression(C = c, penalty = 'l2')
					logist.fit(X_train[train2], y_train[train2])
					self.train_scores[idx_cv_1, idx_c, idx_cv_2] = logist.score(X_train[train2], y_train[train2])
					self.test_scores[idx_cv_1, idx_c, idx_cv_2] = logist.score(X_train[test2], y_train[test2])

			self.best_c_vec[idx_cv_1] = c_range[np.argmax(deneme.test_scores[idx_cv_1,:,:].mean(1))]
			logist = linear_model.LogisticRegression(C = self.best_c_vec[idx_cv_1], penalty = 'l2')
			logist.fit(X_train, y_train)
			self.predictions.extend(logist.predict(X_test))


		# self.kf_total = cross_validation.KFold(len(self.X), n_folds=10, indices=True, shuffle=True, random_state=4)
		# logist = linear_model.LogisticRegression()
		# c_range = np.logspace(0, 4, 10)
		# self.lrgs = grid_search.GridSearchCV(estimator=logist, param_grid=dict(C=c_range), n_jobs=1, scoring = 'roc_auc')
		# log_elastic_net = linear_model.SGDClassifier(loss='log')
		# l1_range = np.linspace(0, 1, 20)
		# alpha_range = [0.0001 , 0.001, 0.01, 0.1, 1]
		# self.lrgs = grid_search.GridSearchCV(estimator=log_elastic_net, param_grid=dict( alpha = alpha_range,
		#                                                                            l1_ratio = l1_range), n_jobs=1)

		# acc_per_fold = [self.lrgs.fit(self.X[train],self.y[train]).score(self.X[test],self.y[test]) 
		#  																	for train, test in self.kf_total]

	def score(self):

		self.auc = roc_auc_score(self.y, np.array(self.predictions))
		print 'Area under ROC is: ', self.auc, 'for subject', self.my_subject
		self.acc = accuracy_score(self.y, np.array(self.predictions))
		print 'Accuracy score is: ', self.acc, 'for subject', self.my_subject
		# self.prediction_accuracy = self.lrgs.score(self.X_test,self.y_test)
		# print 'Prediction accuracy is ', self.prediction_accuracy

		# y_predict = self.lrgs.predict(self.X_test)
		# self.auc = roc_auc_score(self.y_test, y_predict)
		# print 'Area under ROC is: ', self.auc

my_subject = subjects[0]

deneme = strum_second_layer(my_subject, ['EEG-Stim', 'EEG-Cue','Pupil'])
deneme.fit()
deneme.score()

# for idx_cv_1 in range(10):

# 	f, ax = plt.subplots(figsize=(12,8))
# 	c_range = np.logspace(-4, 4, 200)
# 	ax.semilogx(c_range, deneme.test_scores[idx_cv_1,:,:].mean(1), lw=4, c='g', label='test score')
# 	ax.semilogx(c_range, deneme.train_scores[idx_cv_1,:,:].mean(1), lw=4, c='b', label='train score')


# 	ax.fill_between(c_range, deneme.train_scores[idx_cv_1,:,:].min(1), deneme.train_scores[idx_cv_1,:,:].max(1), color = 'b', alpha=0.2)
# 	ax.fill_between(c_range, deneme.test_scores[idx_cv_1,:,:].min(1), deneme.test_scores[idx_cv_1,:,:].max(1), color = 'g', alpha=0.2)

# 	ax.set_ylabel("score for LR",fontsize=16)
# 	ax.set_xlabel("C",fontsize=16)
# 	best_c = c_range[np.argmax(deneme.test_scores[idx_cv_1,:,:].mean(1))]
# 	best_score = deneme.test_scores[idx_cv_1,:,:].mean(1).max()
# 	ax.text(best_c, best_score+0.05, "C = %6.4f | score=%6.4f" % (best_c, best_score),\
# 	        fontsize=15, bbox=dict(facecolor='w',alpha=0.5))
# 	[x.set_fontsize(16) for x in ax.xaxis.get_ticklabels()]
# 	[x.set_fontsize(16) for x in ax.yaxis.get_ticklabels()]
# 	ax.legend(fontsize=16,  loc=0)
# 	ax.set_ylim(0, 1.1)
# 	ax.set_title('Fold ' + str(idx_cv_1))
# 	plt.show()

#pylab.title('Minimal Energy Configuration of %s Charges on Disc W = %s'%(N, W))

