#!/usr/bin/env python
# -*- coding=utf-8 -*-

"""
Develops a model on the training data and tests it on the testing data to understand whether an unknown speech
can be correctly classified according to the party the speaker belongs to.
"""

import unicodedata
import pickle
import regex as re
import pandas as pd
from pandas import *
import numpy as np
import collections
import math
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from processing_functions import compute_tfidf
from xgboost import XGBClassifier


train_cols = []

# Develops the training model
def run_train_classification(bigram_norobespierre_speeches, unigram_norobespierre_speeches, bigram_norobespierre_freq, unigram_norobespierre_freq, bigram_norobespierre_doc_freq, unigram_norobespierre_doc_freq, num_speeches):
	iteration = "train"
	train, train_classification, speeches, speakers = data_clean(iteration, None, bigram_norobespierre_speeches, unigram_norobespierre_speeches, bigram_norobespierre_freq, unigram_norobespierre_freq, bigram_norobespierre_doc_freq, unigram_norobespierre_doc_freq, num_speeches)

	##### There are two models - Logistic Regression and XGB

	### Logistic Regression
	model = LogisticRegression(penalty = 'l1')
	model.fit(train.get_values(), train_classification)
	cv_scores = cross_val_score(model, train.get_values(), train_classification, cv = 10)

	### xgboost
	"""model = XGBClassifier()
	model.fit(train.get_values(), train_classification)
	cv_scores = cross_val_score(model, train.get_values(), train_classification, cv = 10)"""
	
	print "Training CV Score: %f" % np.mean(cv_scores)


	#print ("Training CV Score: " + str(metrics.accuracy_score(train_classification, predicted)))

	train_classification = pd.DataFrame(train_classification)
	speeches = pd.DataFrame(speeches, columns = ['Speechid'])
	speakers = pd.DataFrame(speakers, columns = ['Speaker'])
	
	# Creates a comprehensive dataframe to write to excel
	train_total = pd.concat([train, train_classification, speeches, speakers], axis = 1)
	writer = pd.ExcelWriter("training_data.xlsx")
	train_total.to_excel(writer, 'Sheet1')
	writer.save()

	# Deletes these variables from memory to conserve space for other computations
	train_classification = None
	speeches = None
	speakers = None
	train_total = None
	columns_to_return = train.columns
	train = None


	return [model, columns_to_return]

# Runs the training model on the test set and generates predictions
def run_test_classification(model, train_columns, bigram_norobespierre_speeches, unigram_norobespierre_speeches, bigram_norobespierre_freq, unigram_norobespierre_freq, bigram_norobespierre_doc_freq, unigram_norobespierre_doc_freq, num_speeches):
	iteration = "test"

	test, test_classification, speeches, speakers = data_clean(iteration, train_columns, bigram_norobespierre_speeches, unigram_norobespierre_speeches, bigram_norobespierre_freq, unigram_norobespierre_freq, bigram_norobespierre_doc_freq, unigram_norobespierre_doc_freq, num_speeches)

	# Add columns of zero for features not in the test set but in the training set
	col_names = []
	for col in train_columns:
		if col not in test.columns:
			test[col] = 0
			col_names.append(col)

	# Removes/ignores any columns not in the training data
	test = test[train_columns]

	print "Test CV Score: " + str(model.score(test.get_values(), test_classification))

	# Creates a comprehensive dataframe
	test_classification_df = pd.DataFrame(test_classification, columns = ['Real classification'])
	speeches = pd.DataFrame(speeches, columns = ['Speechid'])
	speakers = pd.DataFrame(speakers, columns = ['Speaker'])

	# Generates predictions
	predictions = model.predict(test.get_values())
	predicted_values = pd.DataFrame(predictions, columns = ['Predicted'])
	write_to = pd.ExcelWriter("predictions.xlsx")
	#real_pred.to_excel(write_to, 'Sheet1')
	write_to.save()

	# Develops and prints the confusion matrix
	print confusion_matrix(test_classification, predicted_values)

	# Generates prediction probabilities
	predict_prob = pd.DataFrame(model.predict_proba(test.get_values()), columns = ['Prob 0', 'Prob 1'])
	
	# Creates a comprehensive dataframe to develop an Excel file
	real_pred = pd.DataFrame()
	real_pred = pd.concat([test_classification_df, predicted_values, predict_prob, speeches, speakers], axis = 1)

	test_total = pd.concat([test, test_classification_df, speeches, speakers], axis = 1)
	writer = pd.ExcelWriter("test_data.xlsx")
	test_total.to_excel(writer, 'Sheet1')
	writer.save()

	return real_pred

# Cleans the data and creates the feature set
def data_clean(iteration, train_columns, bigram_norobespierre_speeches, unigram_norobespierre_speeches, bigram_norobespierre_freq, unigram_norobespierre_freq, bigram_norobespierre_doc_freq, unigram_norobespierre_doc_freq, num_speeches):
	classification = []
	data_set = []
	speeches = []
	speakers = []

	speechid_to_speaker = pickle.load(open("speechid_to_speaker_store.pickle", "rb"))
	speakers_to_analyze = pickle.load(open("speakers_to_analyze_store.pickle", "rb"))
	### Should I do this once for all the data and then split it into test and train? That way all the data is based on the same bigram_norobespierres. Or is that
	### bad because then the training data is connected to the test data via the tfidf calculations?
	for speechid in bigram_norobespierre_speeches:
		speaker = speechid_to_speaker[speechid]

		#create a vector of speechids in correct order to reverse engineer and check which speeches were/were not correctly classified
		speeches.append(speechid)
		speakers.append(speaker)

		if speakers_to_analyze.loc[speaker, "Party"] == "Girondins":
			classification.append(0)
		else:
			classification.append(1)

		# Feature selection taking place here
		# Analysis accounts for bigram_norobespierres and unigram_norobespierres
		if iteration == "train":
			# Restricting features according to how many times they appear
			bigram_norobespierre_input = {k:v for k,v in bigram_norobespierre_speeches[speechid].items() if (bigram_norobespierre_freq[k] >= 20)}
			unigram_norobespierre_input = {k:v for k,v in unigram_norobespierre_speeches[speechid].items() if (unigram_norobespierre_freq[k] >= 62)}
			
			bigram_norobespierre_scores = compute_tfidf(bigram_norobespierre_input, num_speeches, bigram_norobespierre_doc_freq)
			unigram_norobespierre_scores = compute_tfidf(unigram_norobespierre_input, num_speeches, unigram_norobespierre_doc_freq)
		else:
			bigram_norobespierre_input = {k:v for k,v in bigram_norobespierre_speeches[speechid].items() if (k in train_columns)}
			unigram_norobespierre_input = {k:v for k,v in unigram_norobespierre_speeches[speechid].items() if (k in train_columns)}
			
			bigram_norobespierre_scores = compute_tfidf(bigram_norobespierre_input, num_speeches, bigram_norobespierre_doc_freq)
			unigram_norobespierre_scores = compute_tfidf(unigram_norobespierre_input, num_speeches, unigram_norobespierre_doc_freq)
			
		
		merge_scores = bigram_norobespierre_scores.copy()
		merge_scores.update(unigram_norobespierre_scores)
		
		data_set.append(merge_scores)

	# Remove data from memory to clear space for other computations
	speechid_to_speaker = None
	speakers_to_analyze = None
	bigram_norobespierre_speeches = None
	unigram_norobespierre_speeches = None
	bigram_norobespierre_input = None
	unigram_norobespierre_input = None
	bigram_norobespierre_scores = None
	unigram_norobespierre_scores = None

	data = pd.DataFrame(data_set)
	data_set = None
	data = data.fillna(0)

	return([data, classification, speeches, speakers])


if __name__ == '__main__':
	import sys

	# Load relevant stored data
	train_speeches_bigram_norobespierre = pickle.load(open("train_speeches_bigram_norobespierre.pickle", "rb"))
	train_speeches_unigram_norobespierre = pickle.load(open("train_speeches_unigram_norobespierre.pickle", "rb"))
	train_total_freq_bigram_norobespierre = pickle.load(open("train_total_freq_bigram_norobespierre.pickle", "rb"))
	train_total_freq_unigram_norobespierre = pickle.load(open("train_total_freq_unigram_norobespierre.pickle", "rb"))
	bigram_norobespierre_doc_freq = pickle.load(open("bigram_norobespierre_doc_freq.pickle", "rb"))
	unigram_norobespierre_doc_freq = pickle.load(open("unigram_norobespierre_doc_freq.pickle", "rb"))
	train_number_speeches = pickle.load(open("train_number_speeches_norobespierre.pickle", "rb"))


	model, train_columns = run_train_classification(train_speeches_bigram_norobespierre, train_speeches_unigram_norobespierre, train_total_freq_bigram_norobespierre, train_total_freq_unigram_norobespierre, bigram_norobespierre_doc_freq, unigram_norobespierre_doc_freq, train_number_speeches)

	train_speeches_bigram_norobespierre = None
	train_speeches_unigram_norobespierre = None
	train_total_freq_bigram_norobespierre = None
	train_total_freq_unigram_norobespierre = None


	test_total_freq_bigram_norobespierre = pickle.load(open("test_total_freq_bigram_norobespierre.pickle", "rb"))
	test_total_freq_unigram_norobespierre = pickle.load(open("test_total_freq_unigram_norobespierre.pickle", "rb"))
	test_speeches_bigram_norobespierre = pickle.load(open("test_speeches_bigram_norobespierre.pickle", "rb"))
	test_speeches_unigram_norobespierre = pickle.load(open("test_speeches_unigram_norobespierre.pickle", "rb"))


	real_pred = run_test_classification(model, train_columns, test_speeches_bigram_norobespierre, test_speeches_unigram_norobespierre, test_total_freq_bigram_norobespierre, test_total_freq_unigram_norobespierre, bigram_norobespierre_doc_freq, unigram_norobespierre_doc_freq, train_number_speeches)

	with open("real_pred.pickle", 'wb') as handle:
		pickle.dump(real_pred, handle, protocol = 0)

	# Build comprehensive dataset with speeches to see why a speech may have been misclassified
	real_pred = pd.concat([real_pred, pd.DataFrame(columns = ['Speech Text'])])

	raw_speeches = pickle.load(open("raw_speeches.pickle", "rb"))

	# Includes only text of speeches where the speech was misclassified
	for i, index in enumerate(real_pred.index.values):
		if real_pred['Real classification'].iloc[i] != real_pred['Predicted'].iloc[i]:
			real_pred['Speech Text'].iloc[i] = raw_speeches[real_pred['Speechid'].iloc[i]]

	write_to = pd.ExcelWriter("predictions_with_speeches_norobespierre.xlsx")
	real_pred.to_excel(write_to, 'Sheet1')
	write_to.save()
