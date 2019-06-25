#!/usr/bin/env python
# -*- coding=utf-8 -*-

import pickle
import pandas as pd
from pandas import *
import numpy as np
from nltk import word_tokenize
from nltk.util import ngrams
import collections
from collections import Counter
import os
import regex as re
from make_ngrams import compute_ngrams
import math
from collections import defaultdict
from processing_functions import load_list, load_speakerlist, process_excel, remove_diacritic, compute_tfidf, normalize_dicts, write_to_excel, convert_keys_to_string, compute_difference, cosine_similarity
from scipy import spatial

global num_speeches
doc_freq = pickle.load(open("bigram_doc_freq.pickle", "rb"))

# This is the function that reads in the Excel files and calls the necessary functions to compute the distances
# It then writes those distance dictionaries to Excel
def distance_analysis():

	gir_tfidf = process_excel('girondins_tfidf_allbigrams.xlsx')
	mont_tfidf = process_excel("montagnards_tfidf_allbigrams.xlsx")

	gir_dict = convert_keys_to_string(gir_tfidf)
	mont_dict = convert_keys_to_string(mont_tfidf)
	gir_mont_diff = compute_difference(gir_dict, mont_dict)

	by_month = pd.read_excel("By_Month.xlsx")
	by_date = pd.read_excel("By_Date.xlsx")
	by_speaker = pd.read_excel("By_Speaker_Convention.xlsx")

	by_month = create_tfidf_vectors(by_month)
	by_month_dist = compute_distances(by_month, 'month',  gir_dict, mont_dict, gir_mont_diff)
	write_to_excel(by_month_dist, 'by_month_distances.xlsx')

	by_date = create_tfidf_vectors(by_date)
	by_period = aggregate_by_period(by_date)

	by_period_dist = compute_distances(by_period, 'period', gir_dict, mont_dict, gir_mont_diff)
	write_to_excel(by_period_dist, "by_period_distances.xlsx")

	by_date_dist = compute_distances(by_date, 'date',  gir_dict, mont_dict, gir_mont_diff)
	write_to_excel(by_date_dist, 'by_date_distances.xlsx')

	by_speaker = create_tfidf_vectors(by_speaker)
	by_speaker_dist = compute_distances(by_speaker, 'speaker', gir_dict, mont_dict, gir_mont_diff)
	write_to_excel(by_speaker_dist, 'by_speaker_distances.xlsx')

# Aggregates the data based on three periods - before the convention, during the convention, and after the convention
def aggregate_by_period(dataframe):
	before_convention = Counter()
	convention = Counter()
	after_convention = Counter()
	for i, time in enumerate(dataframe['Full Date']):
		# Convert time to a string to do the string equality analysis to determine which period the row belongs to
		time = str(time)
		if (time >= "1792-6-10") and (time <= "1792-8-10"):
			before_convention = before_convention + dataframe['ngrams'].iloc[i]
		if (time >= "1792-9-20") and (time < "1793-6-2"):
			convention = convention + dataframe['ngrams'].iloc[i]
		if (time >= "1793-6-2") and (time <= "1793-8-2"):
			after_convention = after_convention + dataframe['ngrams'].iloc[i]

	before_convention_tfidf = compute_tfidf(before_convention, num_speeches, doc_freq)
	convention_tfidf = compute_tfidf(convention, num_speeches, doc_freq)
	after_convention_tfidf = compute_tfidf(after_convention, num_speeches, doc_freq)

	before_convention_df = pd.DataFrame.from_dict(before_convention_tfidf, orient = "index")
	convention_df = pd.DataFrame.from_dict(convention_tfidf, orient = "index")
	after_convention_df = pd.DataFrame.from_dict(after_convention_tfidf, orient = "index")

	period_df = [before_convention_tfidf, convention_tfidf, after_convention_tfidf]
	return period_df
	
# Creates two new columns in each dataframe - ngram Counter objects and tfidf dictionaries
# These columsn are used for aggregation and cosine similarity computation
def create_tfidf_vectors(dataframe):
	speeches = dataframe['concat_speeches'].tolist()
	ngrams = []
	for unit in speeches:
		ngrams.append(compute_ngrams(unit, 2))
	ngrams_to_add = pd.Series(ngrams)
	dataframe['ngrams'] = ngrams_to_add.values
	tfidf = []
	for element in ngrams:
		tfidf.append(compute_tfidf(element, num_speeches, doc_freq))
	tfidf_to_add = pd.Series(tfidf)
	dataframe['tfidf'] = tfidf_to_add.values
	return dataframe

# This function computes the cosine similarity and distances between the given dataframe and the three points of analysis
# It assumes that the dataframe contains a tfidf column
def compute_distances(dataframe, period, gir_dict, mont_dict, gir_mont_diff):
	period_vector = []
	if period == 'month':
		period_vector = dataframe['Year-Month'].tolist()
		period_vector = pd.Series(period_vector)
		tfidf_scores = dataframe['tfidf'].tolist()
	elif period == 'date':
		period_vector = dataframe['Date'].tolist()
		period_vector = pd.Series(period_vector)
		tfidf_scores = dataframe['tfidf'].tolist()
	elif period == 'speaker':
		period_vector = dataframe['Speaker'].tolist()
		period_vector = pd.Series(period_vector)
		tfidf_scores = dataframe['tfidf'].tolist()
	else:
		periods = ["Before convention", "Convention", "After convention"]
		period_vector = pd.Series(periods)
		# This assumes that tfidf_scores for the periods is a list not a pandas dataframe
		tfidf_scores = dataframe

	gir_dist = []
	mont_dist = []
	gir_mont_diff_dist = []
	# This for loop is contingent on tfidf_scores being a list
	for counter in tfidf_scores:
		to_compare = convert_keys_to_string(counter)
		# Checks if there tfidf_scores vector exists. If it doesn't, default values are assigned for the distance
		# This was particularly relevant as there was a speaker with tfidf_scores of length 0
		if len(to_compare) > 0:
			gir_dist.append(1 - cosine_similarity(gir_dict, to_compare))
			mont_dist.append(1 - cosine_similarity(mont_dict, to_compare))
			gir_mont_diff_dist.append(cosine_similarity(gir_mont_diff, to_compare))
		else:
			gir_dist.append(1)
			mont_dist.append(1)
			gir_mont_diff_dist.append(0)

	# Merges the distance lists and creates a comprehensive dataframe to return
	gir_dist = pd.Series(gir_dist)
	mont_dist = pd.Series(mont_dist)
	gir_mont_diff_dist = pd.Series(gir_mont_diff_dist)
	comp_df = pd.DataFrame([period_vector, gir_dist, mont_dist, gir_mont_diff_dist])
	comp_df = comp_df.transpose()
	comp_df.columns = [period, 'distance to gir', 'distance to mont', 'distance to diff']
	return comp_df



if __name__ == '__main__':
    import sys
    file = open('num_speeches.txt', 'r')
    num_speeches = int(file.read())

    distance_analysis()