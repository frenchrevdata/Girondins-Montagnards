#!/usr/bin/env python
# -*- coding=utf-8 -*-

from bs4 import BeautifulSoup
import unicodedata
import os
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
from collections import defaultdict
import os
import gzip
from make_ngrams import compute_ngrams
import math


def remove_diacritic(input):
    '''
    Accept a unicode string, and return a normal string (bytes in Python 3)
    without any diacritical marks.
    '''
    return unicodedata.normalize('NFKD', input).encode('ASCII', 'ignore')


def aggregate_by_group(speakers_to_analyze, Girondins, Montagnards):
	files = os.listdir("../Speakers/")
	gir_doc_freq = {}
	mont_doc_freq = {}
	for filename in files:
		with open('../Speakers/' + filename, "r") as f:
			speaker_data = pickle.load(f)
		#speaker_data = pickle.load(open('../Speakers/' + filename, "r"))
		speaker = re.findall(r'([a-zA-Z\- \']+)_ngrams.pickle', filename)[0]
		party = speakers_to_analyze.loc[speaker, "Party"]
		if party == "Girondins":
			gir_doc_freq = check_num_speakers(speaker_data, gir_doc_freq)
			try:
				Girondins = Girondins + speaker_data
			except NameError:
				Girondins = speaker_data
		else:
			mont_doc_freq = check_num_speakers(speaker_data, mont_doc_freq)
			try:
				Montagnards = Montagnards + speaker_data
			except NameError:
				Montagnards = speaker_data

	Girondins = {k:v for k,v in Girondins.items() if (v >= 3) and (gir_doc_freq[k] > 2)}
	print_to_csv(Girondins, "Girondins_counts.csv")

	Montagnards = {k:v for k,v in Montagnards.items() if (v >= 3) and (mont_doc_freq[k] > 2)}
	print_to_csv(Montagnards, "Montagnards_counts.csv")

	df = pd.DataFrame([Girondins, Montagnards])
	df = df.transpose()
	writer = pd.ExcelWriter('combined_frequency.xlsx')
	df.to_excel(writer, 'Sheet1')
	writer.save()

	gir_frequency = process_excel("Girondins_frequency.xlsx")
	mont_frequencey = process_excel("Montagnards_frequency.xlsx")
	prior = process_excel("prior.xlsx")

	
	computelogpostodds(gir_frequency, mont_frequencey, prior)

	compute_distance(Girondins, Montagnards)

def check_num_speakers(speech_data, party_dict):
	for bigram in speech_data:
		if bigram in party_dict:
			party_dict[bigram] = party_dict[bigram] + 1
		else:
			party_dict[bigram] = 1
	return party_dict

def process_excel(filename):
	xls = ExcelFile(filename)
	first = xls.parse(xls.sheet_names[0])
	first = first.set_index('Bigrams')
	first = first.fillna(0)
	second = first.to_dict()
	for entry in second:
		third = second[entry]
	for item in third:
		third[item] = int(third[item])
	return(third)
	
	
def computelogpostodds(dict1, dict2, prior):
	sigmasquared = defaultdict(float)
	sigma = defaultdict(float)
	delta = defaultdict(float)

	n1 = sum(dict1.values())
	n2 = sum(dict2.values())
	nprior = sum(prior.values())

	for word in prior.keys():
		l1 = float(dict1[word] + prior[word]) / ((n1 + nprior) - (dict1[word] + prior[word]))
		l2 = float(dict2[word] + prior[word]) / ((n2 + nprior) - (dict2[word] + prior[word]))
		sigmasquared[word] = 1/(float(dict1[word]) + float(prior[word])) + 1/(float(dict2[word]) + float(prior[word]))
		sigma[word] = math.sqrt(sigmasquared[word])
		delta[word] = (math.log(l1) - math.log(l2))/sigma[word]

	writer = pd.ExcelWriter('log_post_odds.xlsx')
	data = pd.DataFrame(delta.items())
	data.to_excel(writer, "Sheet1")
	writer.save()


def compute_distance(Girondins, Montagnards):
	diff_counter = {}

	# Normalize counts
	all_sum = 0
	all_sum = all_sum + sum(Girondins.values()) + sum(Montagnards.values())
	
	for key in Girondins:
		Girondins[key] = float(Girondins[key])/all_sum

	for key in Montagnards:
		Montagnards[key] = float(Montagnards[key])/all_sum

	df = pd.DataFrame([Girondins, Montagnards])
	df = df.transpose()
	writer = pd.ExcelWriter('combined_normalized.xlsx')
	df.to_excel(writer, 'Sheet1')
	writer.save()

	print_to_csv(Girondins, "Girondins_counts_normalized.csv")
	print_to_csv(Montagnards, "Montagnards_counts_normalized.csv")
	
	# Compute the Euclidean distance between the two vectors
	## When only bigrams in both groups accounted for
	for bigram in Girondins:
		if bigram in Montagnards:
			diff_counter[bigram] = Girondins[bigram] - Montagnards[bigram]

	sum_of_squares = 0
	for entry in diff_counter:
		sum_of_squares = sum_of_squares + math.pow(diff_counter[entry], 2)
	euclidean_distance = math.sqrt(sum_of_squares)
	print(euclidean_distance)

	## When every bigram accounted for
	"""for bigram in Montagnards:
		if bigram in Girondins:
			Montagnards[bigram] = Girondins[bigram] - Montagnards[bigram]
	for bigram in Girondins:
		if bigram not in Montagnards:
			Montagnards[bigram] = Girondins[bigram]

	sum_of_squares = 0
	for entry in Montagnards:
		sum_of_squares = sum_of_squares + math.pow(Montagnards[entry], 2)
	euclidean_distance = math.sqrt(sum_of_squares)
	print(euclidean_distance)"""

def print_to_csv(dictionary, filename):
	output_file = filename
	with open(filename, mode='w') as f:
		f.write('Bigrams|freq\n')
		for bigram, count in dictionary.items():
			f.write('{}|{}\n'.format(bigram, count))

def load_list(speakernames):
	pd_list = pd.read_excel(speakernames, sheet_name= 'Sheet1')
	pd_list = pd_list.set_index('Name')
	speakers = pd_list.index.tolist()
	for speaker in speakers:
		ind = speakers.index(speaker)
		speakers[ind] = remove_diacritic(speaker).decode('utf-8')
	pd_list.index = speakers
	return pd_list

if __name__ == '__main__':
    import sys
    Girondins = Counter()
    Montagnards = Counter()
    speakers_to_analyze = load_list("Copy of Girondins and Montagnards.xlsx")
    aggregate_by_group(speakers_to_analyze, Girondins, Montagnards)
