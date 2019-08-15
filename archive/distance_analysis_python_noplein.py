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
import csv
import regex as re
from make_ngrams import compute_ngrams
import math
from collections import defaultdict
from processing_functions import load_list, load_speakerlist, process_excel, remove_diacritic, compute_tfidf, normalize_dicts, normalize_by_speeches, write_to_excel, convert_keys_to_string, compute_difference, compute_difference_withplein, cosine_similarity
from scipy import spatial

global num_speeches

doc_freq = pickle.load(open("bigram_doc_freq_noplein.pickle", "rb"))

# This is the function that reads in the Excel files and calls the necessary functions to compute the distances
# It then writes those distance dictionaries to Excel
def distance_analysis():

	gir_tfidf = process_excel('girondins_tfidf_allbigrams.xlsx')
	mont_tfidf = process_excel("montagnards_tfidf_allbigrams.xlsx")

	gir_dict = convert_keys_to_string(gir_tfidf)
	mont_dict = convert_keys_to_string(mont_tfidf)
	gir_mont_diff = compute_difference(gir_dict, mont_dict)
	print "here2"

	#by_month = pickle.load(open("byyearmonth.pickle", "rb"))
	#by_date = pickle.load(open("byfulldate.pickle", "rb"))
	by_speaker = pickle.load(open("byspeaker.pickle", "rb"))
	#by_speaker_allspeakers = pickle.load(open("byspeaker_allspeakers.pickle", "rb"))

	"""by_month = create_tfidf_vectors(by_month)
	by_month_dist = compute_distances(by_month, 'aggregation',  gir_dict, mont_dict, plein_dict, gir_mont_diff)
	write_to_excel(by_month_dist, 'by_month_distances.xlsx')

	by_period = aggregate_by_period(by_date)
	by_date = create_tfidf_vectors(by_date)

	by_period_dist = compute_distances(by_period, 'period', gir_dict, mont_dict, plein_dict, gir_mont_diff)
	write_to_excel(by_period_dist, "by_period_distances.xlsx")

	by_date_dist = compute_distances(by_date, 'aggregation',  gir_dict, mont_dict, plein_dict, gir_mont_diff)
	write_to_excel(by_date_dist, 'by_date_distances.xlsx')"""

	#by_speaker = create_tfidf_vectors(by_speaker)
	by_speaker_dist = compute_distances(by_speaker, 'speaker', gir_dict, mont_dict, gir_mont_diff)
	write_to_excel(by_speaker_dist, 'by_speaker_noplein_distances_speaker_withsub.xlsx')

	"""by_speaker_allspeakers = create_tfidf_vectors(by_speaker_allspeakers)
	by_speaker_allspeakers_dist = compute_distances(by_speaker_allspeakers, 'speaker', gir_dict, mont_dict, plein_dict, gir_mont_diff)
	write_to_excel(by_speaker_allspeakers_dist, 'by_speaker_allspeakers_distances.xlsx')"""


# Aggregates the data based on three periods - before the convention, during the convention, and after the convention
def aggregate_by_period(dataframe):
	before_convention = Counter()
	convention = Counter()
	after_convention = Counter()
	for key in dataframe:
		time = key
		if (time >= "1792-06-10") and (time <= "1792-08-10"):
			before_convention = before_convention + dataframe[time]
		if (time >= "1792-09-20") and (time < "1793-06-02"):
			convention = convention + dataframe[time]
		if (time >= "1793-06-02") and (time <= "1793-08-02"):
			after_convention = after_convention + dataframe[time]

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
	tfidf = {}
	for element in dataframe:
		tfidf[element] = compute_tfidf(dataframe[element], num_speeches, doc_freq)
	return tfidf

# This function computes the cosine similarity and distances between the given dataframe and the three points of analysis
# It assumes that the dataframe contains a tfidf column
def compute_distances(dataframe, period, gir_dict, mont_dict, gir_mont_diff):
	period_vector = []
	if (period == 'aggregation') or (period == 'speaker'):
		period_vector = list(dataframe.keys())
		period_vector = pd.Series(period_vector)
		speaker_num_speeches = pickle.load(open("speaker_num_speeches.pickle", "rb"))
		"""period_vector = pd.Series(period_vector)
		tfidf_scores = dataframe['tfidf'].tolist()"""
	else:
		periods = ["Before convention", "Convention", "After convention"]
		period_vector = pd.Series(periods)
		# This assumes that tfidf_scores for the periods is a list not a pandas dataframe

	gir_dist = []
	mont_dist = []
	gir_mont_diff_dist = []
	# This for loop is contingent on tfidf_scores being a list
	for element in dataframe:
		"""print type(element)
		print type(dataframe[element])
		to_compare = dataframe[element]"""
		if period == 'speaker':
			#gir = pickle.load(open("Girondins.pickle", "rb"))
			#mont = pickle.load(open("Montagnards.pickle", "rb"))

			# Consider dividing by number of speeches, to normalize
			# Maintain num of speeches per group and number of chars per group

			gir = pickle.load(open("Girondins.pickle", "rb"))
			mont = pickle.load(open("Montagnards.pickle", "rb"))

			file = open('gir_speeches_noplein.txt', 'r')
			gir_num_speeches = int(file.read())

			file = open('mont_speeches_noplein.txt', 'r')
			mont_num_speeches = int(file.read())

			speakers_to_analyze = load_list("Girondins and Montagnards New Mod.xlsx")
			party = speakers_to_analyze.loc[element, "Party"]

			print element
			print party
			print type(dataframe[element])

			if party == 'Girondins':
				gir = gir - dataframe[element]
			if party == 'Montagnards':
				mont = mont - dataframe[element]

			# Normalizing by number of speeches
			#gir_normalized = normalize_by_speeches(gir, gir_num_speeches)
			#mont_normalized = normalize_by_speeches(mont, mont_num_speeches)

			gir_dict = convert_keys_to_string(compute_tfidf(gir, num_speeches, doc_freq))
			mont_dict = convert_keys_to_string(compute_tfidf(mont, num_speeches, doc_freq))
			gir_mont_diff = compute_difference_withplein(gir_dict, mont_dict)

			# Resets the Gir and Mont vectors to their unnormalized version
			#gir_dict_unnormalized = convert_keys_to_string(compute_tfidf(gir, num_speeches, doc_freq))
			#mont_dict_unnormalized = convert_keys_to_string(compute_tfidf(mont, num_speeches, doc_freq))

			w = csv.writer(open("gir_mont_diff.csv", "w"))
			for key, val in gir_mont_diff.items():
				w.writerow([key, val])

			# Normalizing the speaker data as well
			#speaker_speeches = speaker_num_speeches[element]
			#speaker_dict = normalize_by_speeches(dataframe[element], speaker_speeches)

			speaker_dict = dataframe[element]

			tfidf_speaker = compute_tfidf(speaker_dict, num_speeches, doc_freq)

			to_compare = convert_keys_to_string(tfidf_speaker)
		elif period == 'aggregation':
			to_compare = convert_keys_to_string(dataframe[element])
		else:
			to_compare = convert_keys_to_string(element)
		# Checks if there tfidf_scores vector exists. If it doesn't, default values are assigned for the distance
		# This was particularly relevant as there was a speaker with tfidf_scores of length 0
		if len(to_compare) > 0:
			#Normalized
			gir_dist.append(1 - cosine_similarity(gir_dict, to_compare))
			mont_dist.append(1 - cosine_similarity(mont_dict, to_compare))

			#Unnormalized
			#gir_dist.append(1 - cosine_similarity(gir_dict_unnormalized, to_compare))
			#mont_dist.append(1 - cosine_similarity(mont_dict_unnormalized, to_compare))

			gir_mont_diff_dist.append(cosine_similarity(gir_mont_diff, to_compare))
		else:
			gir_dist.append(1)
			mont_dist.append(1)
			gir_mont_diff_dist.append(0)

	# Merges the distance lists and creates a comprehensive dataframe to return
	mont_dist = pd.Series(mont_dist)
	gir_dist = pd.Series(gir_dist)
	gir_mont_diff_dist = pd.Series(gir_mont_diff_dist)
	comp_df = pd.DataFrame([period_vector, gir_dist, mont_dist, gir_mont_diff_dist])
	comp_df = comp_df.transpose()
	comp_df.columns = [period, 'distance to gir', 'distance to mont', 'distance to diff']
	return comp_df



if __name__ == '__main__':
    import sys
    file = open('num_speeches_noplein.txt', 'r')
    num_speeches = int(file.read())

    distance_analysis()
