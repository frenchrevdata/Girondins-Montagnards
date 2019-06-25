#!/usr/bin/env python
# -*- coding=utf-8 -*-

"""
Aggregates the Girondins and Montagnards data, while keeping track of various metrics and data relationships
in order to run classification and do predictive analysis.
"""

from bs4 import BeautifulSoup
import unicodedata
import csv
import pickle
import regex as re
import pandas as pd
from pandas import *
import numpy as np
from nltk import word_tokenize
from nltk.util import ngrams
import collections
from collections import Counter
import os
from make_ngrams import compute_ngrams
import math
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from processing_functions import write_to_excel, load_list, process_excel, remove_diacritic, compute_tfidf, normalize_dicts
from lr_classification import run_train_classification, run_test_classification

date_regex = '([0-9]{4}-[0-9]{2}-[0-9]{1,2})'


def aggregate(speakers_to_analyze, raw_speeches, speechid_to_speaker):
	speaker_names = set()
	speakers_to_consider = []

	# Initialize various data frames for export to the classification script
	train_total_freq_unigram = {}
	test_total_freq_unigram = {}
	train_total_freq_bigram = {}
	test_total_freq_bigram = {}
	train_number_speeches = 0
	test_number_speeches = 0
	# Keeps track of which speeches contain the given bigram
	train_speeches_bigram = collections.defaultdict(dict)
	test_speeches_bigram = collections.defaultdict(dict)
	train_speeches_unigram = collections.defaultdict(dict)
	test_speeches_unigram = collections.defaultdict(dict)

	bigrams_to_speeches = collections.defaultdict()
	bigram_doc_freq = collections.defaultdict()
	unigram_doc_freq = collections.defaultdict()

	gir_num_speeches = 0
	mont_num_speeches = 0
	gir_docs = {}
	mont_docs = {}

	for speaker in speakers_to_analyze.index.values:
		speakers_to_consider.append(remove_diacritic(speaker).decode('utf-8'))

	for speaker_name in speakers_to_consider:
		print speaker_name
		party = speakers_to_analyze.loc[speaker_name, "Party"]
		speech = Counter()
		speech_num = 0
		for identity in raw_speeches:
			date = re.findall(date_regex, str(identity))[0]
			if (date >= "1792-09-20") and (date <= "1793-06-02") and (speaker_name == speechid_to_speaker[identity]):
				# Only looking at speeches with substance, so greater than 100 characters
				if len(raw_speeches[identity]) >= 100:
					indv_speech_bigram = compute_ngrams(raw_speeches[identity], 2)
					indv_speech_unigram = compute_ngrams(raw_speeches[identity], 1)
					# Splitting the data into training and test data with 1/4 of each speaker's data in the test set
					if speech_num%4 != 0:
						train_number_speeches += 1
						for bigram in indv_speech_bigram:
							augment(bigram_doc_freq, bigram)
							augment(train_total_freq_bigram, bigram)
						for unigram in indv_speech_unigram:
							augment(unigram_doc_freq, unigram)
							augment(train_total_freq_unigram, unigram)
						train_speeches_bigram[identity] = indv_speech_bigram
						train_speeches_unigram[identity] = indv_speech_unigram
					else:
						test_number_speeches += 1
						for bigram in indv_speech_bigram:
							augment(test_total_freq_bigram, bigram)
						for unigram in indv_speech_unigram:
							augment(test_total_freq_unigram, unigram)
						test_speeches_bigram[identity] = indv_speech_bigram
						test_speeches_unigram[identity] = indv_speech_unigram

					speech_num += 1
		
	# Write all relevant data objects and values to memory to use when running classification
	with open("speechid_to_speaker_store.pickle", 'wb') as handle:
		pickle.dump(speechid_to_speaker, handle, protocol = 0)
	speechid_to_speaker = None
	with open("speakers_to_analyze_store.pickle", 'wb') as handle:
		pickle.dump(speakers_to_analyze, handle, protocol = 0)
	speakers_to_analyze = None
	raw_speeches = None

	with open("train_speeches_bigram.pickle", 'wb') as handle:
		pickle.dump(train_speeches_bigram, handle, protocol = 0)
	with open("train_speeches_unigram.pickle", 'wb') as handle:
		pickle.dump(train_speeches_unigram, handle, protocol = 0)
	with open("train_total_freq_bigram.pickle", 'wb') as handle:
		pickle.dump(train_total_freq_bigram, handle, protocol = 0)
	with open("train_total_freq_unigram.pickle", 'wb') as handle:
		pickle.dump(train_total_freq_unigram, handle, protocol = 0)
	
	with open("bigram_doc_freq.pickle", 'wb') as handle:
		pickle.dump(bigram_doc_freq, handle, protocol = 0)
	with open("unigram_doc_freq.pickle", 'wb') as handle:
		pickle.dump(unigram_doc_freq, handle, protocol = 0)
	with open("train_number_speeches.pickle", 'wb') as handle:
		pickle.dump(train_number_speeches, handle, protocol = 0)

	with open("test_speeches_bigram.pickle", 'wb') as handle:
		pickle.dump(test_speeches_bigram, handle, protocol = 0)
	with open("test_speeches_unigram.pickle", 'wb') as handle:
		pickle.dump(test_speeches_unigram, handle, protocol = 0)
	with open("test_total_freq_bigram.pickle", 'wb') as handle:
		pickle.dump(test_total_freq_bigram, handle, protocol = 0)
	with open("test_total_freq_unigram.pickle", 'wb') as handle:
		pickle.dump(test_total_freq_unigram, handle, protocol = 0)

# Augments a relevant dictionary to keep track of counts of various bigrams and unigrams
def augment(dictionary, ngram):
	if ngram in dictionary:
		dictionary[ngram] = dictionary[ngram] + 1
	else:
		dictionary[ngram] = 1


if __name__ == '__main__':
    import sys
    raw_speeches = pickle.load(open("raw_speeches.pickle", "rb"))
    speechid_to_speaker = pickle.load(open("speechid_to_speaker.pickle", "rb"))
    speakers_to_analyze = load_list("Girondins and Montagnards New Mod Limit No Robespierre.xlsx")
    aggregate(speakers_to_analyze, raw_speeches, speechid_to_speaker)
