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
from processing_functions import load_list, load_speakerlist, process_excel, remove_diacritic, compute_tfidf, normalize_dicts, write_to_excel
from scipy import spatial

date_regex = '([0-9]{4}-[0-9]{2}-[0-9]{1,2})'

def build_vectors(raw_speeches, speechid_to_speaker, speaker_list, speakers_to_analyze, gir_tfidf, mont_tfidf, num_speeches, doc_freq):
	speaker_ngrams = {}
	speakers_to_consider = []
	speaker_distances = collections.defaultdict()
	chronology = collections.defaultdict(dict)

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
		"""
		if speaker_name in chronology:
			pairing = chronology[speaker_name]
			for bigram in indv_speech_bigram:
				if bigram in pairing:
					pairing[bigram].append([identity, indv_speech_bigram[bigram]])
				else:
					pairing[bigram] = [identity, indv_speech_bigram[bigram]]
		else:
			chronology[speaker_name] = {}
			pairing = chronology[speaker_name]
			for bigram in indv_speech_bigram:
				pairing[bigram] = []
				# stores the unique speechid alongside the number of times that bigram is said in that speech for each bigram
				pairing[bigram] = [identity, indv_speech_bigram[bigram]]"""

	
	## Need tf-idf vectors for gir and mont
	## Need the doc_freq for the previous calcuations
	## compute tf-idf for individual speakers
	## compute cosine distance based on those vectors (dot product over length of vectors)
	## compute cosine similarity between the difference between the two group vectors (subtract from each other)
	## A - B, if positive more like A, if negative more like B

	## create tf vector for each speech and store that so can just add
	## Separately store single idf vector

	#########

	gir_dict = convert_keys_to_string(gir_tfidf)
	mont_dict = convert_keys_to_string(mont_tfidf)
	doc_freq_dict = convert_keys_to_string(doc_freq)
	gir_mont_diff = compute_difference(gir_dict, mont_dict)
	#gir_dict = gir_tfidf
	#print gir_dict
	#mont_dict = mont_tfidf
	for speaker in speaker_ngrams:
		speaker_dict = convert_keys_to_string(speaker_ngrams[speaker])
		to_compare = compute_tfidf(speaker_dict, num_speeches, doc_freq_dict)
		gir_dist = cosine_similarity(gir_dict, to_compare)
		mont_dist = cosine_similarity(mont_dict, to_compare)
		# Need to actually compute the distance
		gir_mont_diff_dist = cosine_similarity(gir_mont_diff, to_compare)
		speaker_distances[speaker] = [gir_dist, mont_dist, gir_mont_diff_dist]

	"""
	#speaker_dict = {(str(k),v) for k,v in speaker_ngrams['Francois Chabot']}
	speaker_dict = convert_keys_to_string(speaker_ngrams['Francois Chabot'])
	to_compare = compute_tfidf(speaker_dict, num_speeches, doc_freq)
	gir_dist = cosine_similarity(gir_dict, to_compare)
	df = pd.DataFrame([to_compare, gir_dict])
	df = df.transpose()
	write_to_excel(df, "Francois Chabot Test.xlsx")"""

	
	"""for speaker in speaker_ngrams:
		#to_compare = {k:v for k,v in speaker_ngrams[speaker].items() if (v >= 3)}
		to_compare = speaker_ngrams[speaker]
		gir_dict = gir_tfidf
		mont_dict = mont_tfidf
		gir_normalized = normalize_dicts(to_compare, gir_dict)
		gir_dist = 	compute_distance(gir_normalized[0], gir_normalized[1])
		to_compare = speaker_ngrams[speaker]
		mont_normalized = normalize_dicts(to_compare, mont_dict)
		mont_dist = compute_distance(mont_normalized[0], mont_normalized[1])
		speaker_distances[speaker] = [gir_dist, mont_dist]"""

	

	
	pickle_filename_3 = "speaker_ngrams.pickle"
	with open(pickle_filename_3, 'wb') as handle:
		pickle.dump(speaker_ngrams, handle, protocol = 0)

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

	"""df2 = pd.DataFrame.from_dict(chronology)
	df2 = df2.transpose()
	filename2 = "chronology.xlsx"
	writer2 = pd.ExcelWriter(filename)
	df2.to_excel(writer2, 'Sheet1')
	writer2.save()

	pickle_filename_2 = "chronology.pickle"
	with open(pickle_filename_2, 'wb') as handle:
		pickle.dump(chronology, handle, protocol = 0)"""

def compute_difference(dict1, dict2):
	diff = {}
	for k in dict1:
		diff[k] = dict1[k] - dict2[k]
	return diff 

# Need this function because the keys for one of the dictionaries were not strings so the match check
# in cosine similarity was not working
def convert_keys_to_string(dictionary):
	to_return = {}
	for k in dictionary:
		to_return[str(k)] = dictionary[k]
	return to_return

def cosine_similarity(dict1, dict2):
	numerator = 0
	denom1 = 0
	denom2 = 0
	for k in dict2:
		if k in dict1:
			numerator += dict1[k]*dict2[k]
		denom2 += dict2[k]*dict2[k]
	for v in dict1.values():
		denom1 += v*v
	cos_sim = numerator/math.sqrt(denom1*denom2)
	return cos_sim

def check_num_speakers(speech_data, speaker, party_dict):
	for bigram in speech_data:
		if bigram in party_dict:
			party_dict[bigram].add(speaker)
		else:
			party_dict[bigram] = set()
			party_dict[bigram].add(speaker)
	return party_dict
	

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
    """#frequency bigrams when only bigrams occuring more than 3 times accounted for
    gir_frequency = process_excel('girondins_frequency.xlsx')
    mont_frequency = process_excel("montagnards_frequency.xlsx")"""
    #frequency vectors when all possible bigrams accounted for
    gir_tfidf = process_excel('girondins_tfidf.xlsx')
    mont_tfidf = process_excel("montagnards_tfidf.xlsx")
    #doc_freq = process_excel("doc_freq.xlsx")
    doc_freq = pickle.load(open("bigram_doc_freq.pickle", "rb"))
    file = open('num_speeches.txt', 'r')
    num_speeches = int(file.read())
    speakers_to_analyze = load_list("Girondins and Montagnards New Mod.xlsx")

    build_vectors(raw_speeches, speechid_to_speaker, speaker_list, speakers_to_analyze, gir_tfidf, mont_tfidf, num_speeches, doc_freq)
