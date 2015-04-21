
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model, cross_validation, grid_search

data = pd.read_csv('strum_etas.csv', nrows = 395)

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

my_subject = subjects[0]

class strum_second_layer:

	def __init__(self, my_subject, modality):
		global data
		self.subject = my_subject
		data_train =  data[data.Subject != self.subject ]   
		data_test =  data[data.Subject == self.subject ]     

		self.y = data_train['Label'].values
		self.X = data_train.ix[:,2:].values

		self.y_test = data_test['Label'].values
		self.X_test = data_test.ix[:,2:].values  

	def fit(self):

		kf_total = cross_validation.KFold(len(self.X), n_folds=10, indices=True, shuffle=True, random_state=4)
		log_elastic_net = linear_model.SGDClassifier(loss='log')
		l1_range = np.linspace(0, 1, 20)
		alpha_range = [0.0001 , 0.001, 0.01, 0.1, 1]
		self.lrgs = grid_search.GridSearchCV(estimator=log_elastic_net, param_grid=dict( alpha = alpha_range,
		                                                                           l1_ratio = l1_range), n_jobs=1)

		acc_per_fold = [self.lrgs.fit(self.X[train],self.y[train]).score(self.X[test],self.y[test]) 
																			for train, test in kf_total]
		self.best_params = self.lrgs.best_params_  																	
		self.acc_mean = np.mean(acc_per_fold)
		print 'Average error across folds: ', self.acc_mean

	def predict(self):

		prediction_accuracy = self.lrgs.score(self.X_test,self.y_test)
		print 'Prediction accuracy is ', prediction_accuracy


deneme = strum_second_layer(my_subject)

#data[filter((lambda x: x.startswith('EEG')), c)].values
