#!/usr/bin/env python
# -*- coding=utf-8 -*-

"""
Aggregates raw data based on Girondins or Montagnards classification.
Computes the distance between the Girondins and Montagnards frequency vectors as well as writes data
to files for further analysis in R.
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
from processing_functions import write_to_excel, load_list, process_excel, remove_diacritic, compute_tfidf, normalize_dicts

date_regex = '([0-9]{4}-[0-9]{2}-[0-9]{1,2})'

# Maintains the number of documents a given bigram is spoken in for use with tf-idf
#bigram_doc_freq = defaultdict(lambda: 0)


def aggregate(speakers_to_analyze, raw_speeches, speechid_to_speaker, Girondins, Montagnards, Plein):
	speaker_names = set()
	speaker_num_speeches = {}
	speaker_char_count = {}
	speakers_to_consider = []

	bigrams_to_speeches = collections.defaultdict()
	bigram_doc_freq = collections.defaultdict()

	gir_num_speeches = 0
	mont_num_speeches = 0
	gir_docs = {}
	mont_docs = {}
	plein_docs = {}

	for speaker in speakers_to_analyze.index.values:
		speakers_to_consider.append(remove_diacritic(speaker).decode('utf-8'))

	for speaker_name in speakers_to_consider:
		print speaker_name
		party = speakers_to_analyze.loc[speaker_name, "Party"]
		speech = Counter()
		for identity in raw_speeches:
			date = re.findall(date_regex, str(identity))[0]
			if (date >= "1792-09-20") and (date <= "1793-06-02") and (speaker_name == speechid_to_speaker[identity]):
				# Keeps track of the number of speeches per speaker as well as the number of characters spoken by each speaker
				# To potentially establish a cutoff for analysis purposes
				augment(speaker_num_speeches, speaker_name)
				if speaker_name in speaker_char_count:
					speaker_char_count[speaker_name] += len(raw_speeches[identity])
				else:
					speaker_char_count[speaker_name] = len(raw_speeches[identity])

				indv_speech_bigram = compute_ngrams(raw_speeches[identity], 2)

				for bigram in indv_speech_bigram:
					augment(bigram_doc_freq, bigram)

					# Maintains a list of speeches in which given bigrams are spoken in
					if bigram in bigrams_to_speeches:
						bigrams_to_speeches[bigram].append(identity)
					else:
						bigrams_to_speeches[bigram] = []
						bigrams_to_speeches[bigram].append(identity)

				# Augments the relevant variables according to the party the speaker belongs to
				if party == "Girondins":
					gir_num_speeches += 1
					gir_docs = check_num_speakers(indv_speech_bigram, speaker_name, gir_docs)
					try:
						Girondins = Girondins + indv_speech_bigram
					except NameError:
						Girondins = indv_speech_bigram
				else:
					mont_num_speeches += 1
					mont_docs = check_num_speakers(indv_speech_bigram, speaker_name, mont_docs)
					try:
						Montagnards = Montagnards + indv_speech_bigram
					except NameError:
						Montagnards = indv_speech_bigram
			

				#speech = speech + indv_speech_bigram

		# Stores the bigram Counter object for each individual speaker
		"""pickle_filename = "../Speakers/" + speaker_name + "_ngrams.pickle"
		with open(pickle_filename, 'wb') as handle:
			pickle.dump(speech, handle, protocol = 0)"""
	
	"""# Stores the bigrams_to_speeches document in Excel
	df_bigrams_to_speeches = pd.DataFrame.from_dict(bigrams_to_speeches, orient = "index")
	write_to_excel(df_bigrams_to_speeches, 'bigrams_to_speeches.xlsx')"""
	w = csv.writer(open("bigrams_to_speeches_noplein.csv", "w"))
	for key, val in bigrams_to_speeches.items():
		w.writerrow([key,val])

	# Computes the tf_idf scores for each bigram and for both the Girondins and Montaganards vectors
	num_speeches = gir_num_speeches + mont_num_speeches
	with open('gir_speeches_noplein.txt', 'w') as f:
		f.write('%d' % gir_num_speeches)
	with open('mont_speeches_noplein.txt', 'w') as f:
		f.write('%d' % mont_num_speeches)
	print num_speeches

	with open('speaker_num_speeches_noplein.pickle', 'wb') as handle:
		pickle.dump(speaker_num_speeches, handle, protocol = 0)

	with open('speaker_char_count_noplein.pickle', 'wb') as handle:
		pickle.dump(speaker_num_speeches, handle, protocol = 0)

	w = csv.writer(open("speaker_num_speeches_noplein.csv", "w"))
	for key, val in speaker_num_speeches.items():
		w.writerow([key, val])

	w = csv.writer(open("speaker_char_count_noplein.csv", "w"))
	for key, val in speaker_char_count.items():
		w.writerow([key, val])

	# Write the number of speeches and doc_frequency to memory for use in further analysis
	with open('num_speeches_noplein.txt', 'w') as f:
		f.write('%d' % num_speeches)
	df_doc_freq = pd.DataFrame.from_dict(bigram_doc_freq, orient = "index")
	write_to_excel(df_doc_freq, 'doc_freq.xlsx')

	with open("bigram_doc_freq_noplein.pickle", 'wb') as handle:
		pickle.dump(bigram_doc_freq, handle, protocol = 0)

	with open("Girondins.pickle", 'wb') as handle:
		pickle.dump(Girondins, handle, protocol = 0)
	with open("Montagnards.pickle", 'wb') as handle:
		pickle.dump(Montagnards, handle, protocol = 0)
	gir_tfidf = compute_tfidf(Girondins, num_speeches, bigram_doc_freq)
	mont_tfidf = compute_tfidf(Montagnards, num_speeches, bigram_doc_freq)

	"""with open("gir_tfidf.pickle", 'wb') as handle:
		pickle.dump(gir_tfidf, handle, protocol = 0)
	with open("mont_tfidf.pickle", 'wb') as handle:
		pickle.dump(mont_tfidf, handle, protocol = 0)"""

	# Computes the distance between the tf_idf vectors
	#compute_distance(gir_tfidf, mont_tfidf)

	# Stores the tf_idf vectors
	df_gir_tfidf = pd.DataFrame.from_dict(gir_tfidf, orient = "index")
	#df_gir_tfidf.columns = ['Bigrams', 'tfidf']
	write_to_excel(df_gir_tfidf, 'gir_tfidf.xlsx')
	df_mont_tfidf = pd.DataFrame.from_dict(mont_tfidf, orient = "index")
	#df_mont_tfidf.columns = ['Bigrams', 'tfidf']
	write_to_excel(df_mont_tfidf, 'mont_tfidf.xlsx')


	df_tfidf_combined = pd.DataFrame([gir_tfidf, mont_tfidf])
	df_tfidf_combined = df_tfidf_combined.transpose()
	df_tfidf_combined.columns = ["Girondins", "Montagnards"]
	write_to_excel(df_tfidf_combined, 'combined_tfidf.xlsx')

	# Constrains the analysis of Girondins and Montagnards frequencies if the frequency more 3 and optionally if in a certain number of speeches
	Girondins = {k:v for k,v in Girondins.items() if (v >= 3)} #and (len(gir_docs[k]) > 1)}
	df_girondins = pd.DataFrame.from_dict(Girondins, orient = "index")
	write_to_excel(df_girondins, "Girondins_counts.xlsx")

	Montagnards = {k:v for k,v in Montagnards.items() if (v >= 3)} #and (len(mont_docs[k]) > 1)}
	df_montagnards = pd.DataFrame.from_dict(Montagnards, orient = "index")
	write_to_excel(df_montagnards, "Montagnards_counts.xlsx")

	# Normalizes the vectors and computes the distance between them
	#normalized = normalize_dicts(Girondins, Montagnards)
	#compute_distance(normalized[0], normalized[1])

	# Stores the Girondins and Montagnards frequency vectors in the same document
	df_combined = pd.DataFrame([Girondins, Montagnards])
	df_combined = df_combined.transpose()
	df_combined.columns = ["Girondins", "Montagnards"]
	write_to_excel(df_combined, 'combined_frequency.xlsx')


# Augments the value of a given dictionary with the given bigram as the key
def augment(dictionary, ngram):
	if ngram in dictionary:
		dictionary[ngram] = dictionary[ngram] + 1
	else:
		dictionary[ngram] = 1

# Maintains a database separately for each group to measure how many speakers mention each bigram
def check_num_speakers(speech_data, speaker, party_dict):
	for bigram in speech_data:
		if bigram in party_dict:
			party_dict[bigram].add(speaker)
		else:
			party_dict[bigram] = set()
			party_dict[bigram].add(speaker)
	return party_dict
	

# Computes the distance between two vectors, specifically Girondins and Montagnards vectors in this case
def compute_distance(Girondins, Montagnards):
	diff_counter = {}

	dict1 = Girondins
	dict2 = Montagnards
	
	# Compute the Euclidean distance between the two vectors
	## When only bigrams in both groups accounted for
	for bigram in dict1:
		if bigram in dict2:
			diff_counter[bigram] = dict1[bigram] - dict2[bigram]

	sum_of_squares = 0
	for entry in diff_counter:
		sum_of_squares = sum_of_squares + math.pow(diff_counter[entry], 2)
	euclidean_distance = math.sqrt(sum_of_squares)
	#print(euclidean_distance)
	#print("---------")

	## When every bigram accounted for
	diff_counter = {}
	for bigram in dict2:
		if bigram in dict1:
			diff_counter[bigram] = dict1[bigram] - dict2[bigram]
	for bigram in Girondins:
		if bigram not in dict2:
			diff_counter[bigram] = dict1[bigram]

	sum_of_squares = 0
	for entry in diff_counter:
		sum_of_squares = sum_of_squares + math.pow(diff_counter[entry], 2)
	euclidean_distance = math.sqrt(sum_of_squares)
	#print(euclidean_distance)


if __name__ == '__main__':
    import sys
    raw_speeches = pickle.load(open("raw_speeches.pickle", "rb"))
    speechid_to_speaker = pickle.load(open("speechid_to_speaker.pickle", "rb"))
    speakers_to_analyze = load_list("Girondins and Montagnards New Mod.xlsx")
    Girondins = Counter()
    Montagnards = Counter()
    Plein = Counter()
    try:
    	os.mkdir('../Speakers')
    except OSError:
    	pass
    aggregate(speakers_to_analyze, raw_speeches, speechid_to_speaker, Girondins, Montagnards, Plein)
