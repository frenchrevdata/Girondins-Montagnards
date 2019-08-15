#!/usr/bin/env python
# -*- coding=utf-8 -*-

"""
This file analyzes whether the language of individual speakers is more similar to one party or the other.
It only looks at speakers in the list of Girondins and Montagnards speakers.
"""

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

date_regex = '([0-9]{4}-[0-9]{2}-[0-9]{1,2})'

# This function creates a dictionary of ngrams for each speaker and computes the distance between the tfidf scores for the vectors to Girondins, Montagnards, and a difference vector
def calculate_distances(raw_speeches, speechid_to_speaker, speaker_list, speakers_to_analyze, gir_tfidf, mont_tfidf, num_speeches, doc_freq):
	speaker_ngrams = {}
	speakers_to_consider = []
	speaker_distances = collections.defaultdict()

	for speaker in speakers_to_analyze.index.values:
		speakers_to_consider.append(remove_diacritic(speaker).decode('utf-8'))

	for identity in raw_speeches:
		date = re.findall(date_regex, str(identity))[0]
		speaker_name = speechid_to_speaker[identity]
		if (date >= "1792-09-20") and (date <= "1793-06-02") and (speaker_name in speakers_to_consider):
			indv_speech_bigram = compute_ngrams(raw_speeches[identity], 2)
			if speaker_name in speaker_ngrams:
				speaker_ngrams[speaker_name] = speaker_ngrams[speaker_name] + indv_speech_bigram
			else:
				speaker_ngrams[speaker_name] = indv_speech_bigram

	# Writes the dictionary of ngrams by speaker to memory
	pickle_filename_3 = "speaker_ngrams.pickle"
	with open(pickle_filename_3, 'wb') as handle:
		pickle.dump(speaker_ngrams, handle, protocol = 0)

	# Converts the keys of the objects to strings in order to do similarity/distance calculations
	gir_dict = convert_keys_to_string(gir_tfidf)
	mont_dict = convert_keys_to_string(mont_tfidf)
	doc_freq_dict = convert_keys_to_string(doc_freq)

	# Computes the difference between the girondins and montagnards vectors to get one score of similarity or difference to the two
	gir_mont_diff = compute_difference(gir_dict, mont_dict)

	# Stores these dictionaries to memory
	with open("gir_dict.pickle", 'wb') as handle:
		pickle.dump(gir_dict, handle, protocol = 0)
	with open("mont_dict.pickle", 'wb') as handle:
		pickle.dump(mont_dict, handle, protocol = 0)
	with open("gir_mont_diff.pickle", 'wb') as handle:
		pickle.dump(gir_mont_diff, handle, protocol = 0)

	# Iterates through every speaker and calculates the dist to the Girondins, Montagnards and difference vectors
	for speaker in speaker_ngrams:
		speaker_dict = convert_keys_to_string(speaker_ngrams[speaker])
		to_compare = compute_tfidf(speaker_dict, num_speeches, doc_freq_dict)
		gir_dist = 1 - cosine_similarity(gir_dict, to_compare)
		mont_dist = 1 - cosine_similarity(mont_dict, to_compare)
		# If gir_mont_diff_dist is positive, the speaker is more like the Girondins and if it is negative, the speaker is more like the Montagnards
		gir_mont_diff_dist = cosine_similarity(gir_mont_diff, to_compare)
		speaker_distances[speaker] = [gir_dist, mont_dist, gir_mont_diff_dist]

	# Stores the distances for each speaker in a dictionary and writes that dictionary to Excel and to memory
	df = pd.DataFrame.from_dict(speaker_distances)
	df = df.transpose()
	df.columns = ["dist to Girondins", "dist to Montagnards", "dist to difference"]
	filename = "freq_dist_map.xlsx"
	writer = pd.ExcelWriter(filename)
	df.to_excel(writer, 'Sheet1')
	writer.save()

	pickle_filename = "freq_dist.pickle"
	with open(pickle_filename, 'wb') as handle:
		pickle.dump(speaker_distances, handle, protocol = 0)


# Computes the euclidean distance between two dictionaries of bigram to bigram frequency
def compute_distance(first_dict, second_dict):
	diff_counter = {}
	
	dict1 = first_dict
	dict2 = second_dict
	# Compute the Euclidean distance between the two vectors
	## When only bigrams in both groups accounted for
	"""for bigram in dict1:
		if bigram in dict2:
			diff_counter[bigram] = dict1[bigram] - dict2[bigram]

	sum_of_squares = 0
	for entry in diff_counter:
		sum_of_squares = sum_of_squares + math.pow(diff_counter[entry], 2)
	euclidean_distance = math.sqrt(sum_of_squares)
	return(euclidean_distance)
	#print(euclidean_distance)
	#print("---------")"""

	## When every bigram accounted for
	diff_counter = {}
	for bigram in dict2:
		if bigram in dict1:
			diff_counter[bigram] = dict1[bigram] - dict2[bigram]
	for bigram in dict1:
		if bigram not in dict2:
			diff_counter[bigram] = dict1[bigram]

	sum_of_squares = 0
	for entry in diff_counter:
		sum_of_squares = sum_of_squares + math.pow(diff_counter[entry], 2)
	euclidean_distance = math.sqrt(sum_of_squares)
	return(euclidean_distance)
	#print(euclidean_distance)


if __name__ == '__main__':
    import sys
    raw_speeches = pickle.load(open("raw_speeches.pickle", "rb"))
    speechid_to_speaker = pickle.load(open("speechid_to_speaker.pickle", "rb"))
    speaker_list = load_speakerlist('Copy of AP_Speaker_Authority_List_Edited_3.xlsx')

    # There are two tfidf dictionaries, one which only contains bigrams that appear in more than three speeches, below
    #gir_tfidf = process_excel('girondins_tfidf.xlsx')
    #mont_tfidf = process_excel("montagnards_tfidf.xlsx")

    # The second set of dictionaries contain all bigrams said in every speech
    gir_tfidf = process_excel('girondins_tfidf_allbigrams.xlsx')
    mont_tfidf = process_excel("montagnards_tfidf_allbigrams.xlsx")

    doc_freq = pickle.load(open("bigram_doc_freq.pickle", "rb"))

    file = open('num_speeches.txt', 'r')
    num_speeches = int(file.read())
    speakers_to_analyze = load_list("Girondins and Montagnards New Mod.xlsx")

    calculate_distances(raw_speeches, speechid_to_speaker, speaker_list, speakers_to_analyze, gir_tfidf, mont_tfidf, num_speeches, doc_freq)
