#!/usr/bin/env python
# -*- coding=utf-8 -*-

import pickle
import csv
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

	by_speaker = pickle.load(open("byspeaker.pickle", "rb"))
	speakernumwords = pickle.load(open("speakernumwords.pickle", "rb"))

	speaker_tfidf = {}
	for speaker in by_speaker:
		counter = by_speaker[speaker]
		counter = convert_keys_to_string(counter)
		# Tried doing v>=3 but there are some speakers who do not say any bigrams more than 3 times
		freq = {k:v for k,v in counter.items() if (v >= 2)}
		tfidf = compute_tfidf(freq, num_speeches, doc_freq)
		speaker_tfidf[speaker] = tfidf

	robespierre = speaker_tfidf["Maximilien-Francois-Marie-Isidore-Joseph de Robespierre"]

	speaker_dist = {}

	for speaker in speaker_tfidf:
		if str(speaker) != "Maximilien-François-Marie-Isidore-Joseph de Robespierre":
			print speaker_tfidf[speaker]
			dist = 1-cosine_similarity(robespierre, speaker_tfidf[speaker])
			speaker_dist[speaker] = dist


	w = csv.writer(open("dist_to_robespierre_withlimit.csv", "w"))
	for key, val in speaker_dist.items():
		w.writerow([key,val])


	
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



if __name__ == '__main__':
    import sys
    file = open('num_speeches_noplein_withlimit.txt', 'r')
    num_speeches = int(file.read())

    distance_analysis()
