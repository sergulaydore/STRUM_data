
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model, cross_validation, grid_search
from sklearn.metrics import roc_auc_score, accuracy_score

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
		c_range = np.logspace(-3, 3, 100)
		cv = cross_validation.ShuffleSplit(len(self.X), n_iter=n_iter, test_size=0.4)
		self.train_scores = np.zeros((len(c_range), n_iter))
		self.test_scores = np.zeros((len(c_range), n_iter))

		for idx_c, c in enumerate(c_range):
			for idx_cv, (train, test) in enumerate(cv):
				logist = linear_model.LogisticRegression(C = c, penalty = 'l1')
				logist.fit(self.X[train], self.y[train])
				self.train_scores[idx_c, idx_cv] = logist.score(self.X[train], self.y[train])
				self.test_scores[idx_c, idx_cv] = logist.score(self.X[test], self.y[test])


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

	def predict(self):

		self.predictions = []

		for train, test in self.kf_total:
			# self.lrgs.fit(self.X[train],self.y[train])
			self.predictions.extend(self.lrgs.predict(self.X[test]))

		self.auc = roc_auc_score(self.y, np.array(self.predictions))
		print 'Area under ROC is: ', self.auc
		self.acc = accuracy_score(self.y, np.array(self.predictions))
		print 'Accuracy score is: ', self.acc
		# self.prediction_accuracy = self.lrgs.score(self.X_test,self.y_test)
		# print 'Prediction accuracy is ', self.prediction_accuracy

		# y_predict = self.lrgs.predict(self.X_test)
		# self.auc = roc_auc_score(self.y_test, y_predict)
		# print 'Area under ROC is: ', self.auc


i=0
my_subject = subjects[i]

deneme = strum_second_layer(my_subject, ['EEG-Stim', 'EEG-Cue','Pupil'])
deneme.fit()

from matplotlib import pyplot as plt
f, ax = plt.subplots(figsize=(12,8))
c_range = np.logspace(-3, 3, 100)
#for i in range(n_iter):
#    ax.semilogx(gammas, train_scores[:, i], alpha=0.2, lw=2, c='b')
#    ax.semilogx(gammas, test_scores[:, i], alpha=0.2, lw=2, c='g')
ax.semilogx(c_range, deneme.test_scores.mean(1), lw=4, c='g', label='test score')
ax.semilogx(c_range, deneme.train_scores.mean(1), lw=4, c='b', label='train score')


ax.fill_between(c_range, deneme.train_scores.min(1), deneme.train_scores.max(1), color = 'b', alpha=0.2)
ax.fill_between(c_range, deneme.test_scores.min(1), deneme.test_scores.max(1), color = 'g', alpha=0.2)

ax.set_ylabel("score for LR",fontsize=16)
ax.set_xlabel("C",fontsize=16)
best_c = c_range[np.argmax(deneme.test_scores.mean(1))]
best_score = deneme.test_scores.mean(1).max()
ax.text(best_c, best_score+0.05, "C = %6.4f | score=%6.4f" % (best_c, best_score),\
        fontsize=15, bbox=dict(facecolor='w',alpha=0.5))
[x.set_fontsize(16) for x in ax.xaxis.get_ticklabels()]
[x.set_fontsize(16) for x in ax.yaxis.get_ticklabels()]
ax.legend(fontsize=16,  loc=0)
ax.set_ylim(0, 1.1)
plt.show()

