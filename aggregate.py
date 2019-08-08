#!/usr/bin/env python
# -*- coding=utf-8 -*-

"""
Aggregates raw data based on Girondins or Montagnards classification.
Computes the distance between the Girondins and Montagnards frequency vectors as well as writes data
to files for further analysis in R.
"""

import unicodedata
import csv
import pickle
import regex as re
import pandas as pd
from pandas import *
import numpy as np
import collections
from collections import Counter
import os
from make_ngrams import compute_ngrams
import math
from collections import defaultdict
from processing_functions import write_to_excel, load_list, process_excel, remove_diacritic, compute_tfidf, normalize_dicts, store_to_pickle, write_to_csv

# Regex used to find date of session
date_regex = '([0-9]{4}-[0-9]{2}-[0-9]{1,2})'

def aggregate(speakers_to_analyze, raw_speeches, speechid_to_speaker, Girondins, Montagnards):
	speaker_num_speeches = {}
	speaker_char_count = {}
	
	# Dataframe to keep track of the speakers we care about
	speakers_to_consider = []
	# Reformats speakers_to_analyze by removing accents in order to match speakers to those in raw_speeches
	# and speechid_to_speaker
	for speaker in speakers_to_analyze.index.values:
		speakers_to_consider.append(remove_diacritic(speaker).decode('utf-8'))

	# Matches bigrams to the list of speakers and speeches that have that bigram
	bigrams_to_speeches = {}
	bigrams_to_speakers = {}

	# Maintains the number of documents a given bigram is spoken in for use with tf-idf
	bigram_doc_freq = collections.defaultdict()

	gir_num_speeches = 0
	mont_num_speeches = 0
	gir_docs = {}
	mont_docs = {}

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
					if bigram in bigrams_to_speakers:
						bigrams_to_speakers[bigram].add(speaker_name)
					else:
						bigrams_to_speakers[bigram] = set()
						bigrams_to_speakers[bigram].add(speaker_name)

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
			
				### Maintains a Counter of all the bigrams and their counts for a given speaker
				# speech = speech + indv_speech_bigram

	### Stores the bigram Counter object for each individual speaker
		# pickle_filename = "../Speakers/" + speaker_name + "_ngrams.pickle"
		# with open(pickle_filename, 'wb') as handle:
		# 	pickle.dump(speech, handle, protocol = 0)

	# Store raw counts
	store_to_pickle(Girondins,"Girondins.pickle")
	store_to_pickle(Montagnards, "Montagnards.pickle")

	# Store in memory aggregate information about each bigram
	bigram_aggregate_info(Girondins, Montagnards, bigrams_to_speakers, bigrams_to_speeches)


	### If data has already been stored to memory, the lines below can be used
	# bigrams_to_speakers = pickle.load(open("bigrams_to_speakers.pickle", "rb"))
	# bigrams_to_speeches = pickle.load(open("bigrams_to_speeches.pickle", "rb"))

	# gir_docs = pickle.load(open("gir_docs.pickle", "rb"))
	# mont_docs = pickle.load(open("mont_docs.pickle", "rb"))

	# Girondins = pickle.load(open("Girondins_withlimit.pickle", "rb"))
	# Montagnards = pickle.load(open("Montagnards_withlimit.pickle", "rb"))

	# bigram_doc_freq = pickle.load(open("bigram_doc_freq.pickle", 'rb'))

	num_speeches = 4479

	# Computes counts and tfidf scores for each party and outputs for further analysis in R
	counts_and_tfidf(Girondins, Montagnards, gir_docs, mont_docs, num_speeches, bigram_doc_freq)



	""" EVERYTHING BELOW IS STORING DATA TO MEMORY """
	
	# Stores the bigrams_to_speeches document in Excel
	df_bigrams_to_speeches = pd.DataFrame.from_dict(bigrams_to_speeches, orient = "index")
	write_to_excel(df_bigrams_to_speeches, 'bigrams_to_speeches.xlsx')
	df_bigrams_to_speakers = pd.DataFrame.from_dict(bigrams_to_speakers, orient = "index")
	write_to_excel(df_bigrams_to_speakers, 'bigrams_to_speakers.xlsx')
	df_doc_freq = pd.DataFrame.from_dict(bigram_doc_freq, orient = "index")
	write_to_excel(df_doc_freq, 'doc_freq.xlsx')
	
	# Stores files in memory
	store_to_pickle(bigrams_to_speakers, "bigrams_to_speakers.pickle")
	store_to_pickle(bigrams_to_speeches, "bigrams_to_speeches.pickle")
	store_to_pickle(gir_docs, "gir_docs.pickle")
	store_to_pickle(mont_docs, "mont_docs.pickle")
	store_to_pickle(speaker_num_speeches, "speaker_num_speeches.pickle")
	store_to_pickle(speaker_char_count, "speaker_char_count.pickle")
	store_to_pickle(bigram_doc_freq, "bigram_doc_freq.pickle")

	with open('gir_speeches.txt', 'w') as f:
		f.write('%d' % gir_num_speeches)
	with open('mont_speeches.txt', 'w') as f:
		f.write('%d' % mont_num_speeches)

	write_to_csv(speaker_num_speeches, "speaker_num_speeches.csv")
	write_to_csv(speaker_char_count, "speaker_char_count.csv")

	with open('num_speeches.txt', 'w') as f:
		f.write('%d' % num_speeches)

# Aggregates information about each bigram
def bigram_aggregate_info(Girondins, Montagnards, bigrams_to_speakers, bigrams_to_speeches):
	bigram_num_speakers = []
	bigram_num_speeches = []
	bigram_total_freq = []
	bg_speeches = {}
	bigrams = []
	speeches = []
	speakers = []
	for bigram in bigrams_to_speeches:
		if (Girondins[bigram] >= 10) or (Montagnards[bigram] >= 10):
			bigram_num_speakers.append(len(bigrams_to_speakers[bigram]))
			bigram_num_speeches.append(len(bigrams_to_speeches[bigram]))
			bigram_total_freq.append(Girondins[bigram] + Montagnards[bigram])
			bigrams.append(str(bigram))
			speeches.append(str(bigrams_to_speeches[bigram]))
			speakers.append(str(bigrams_to_speakers[bigram]))

	bg_num_speakers = pd.DataFrame(bigram_num_speakers, columns = ['Num Speakers'])
	bg_num_speeches = pd.DataFrame(bigram_num_speeches, columns = ['Num Speeches'])
	bg_total_freq = pd.DataFrame(bigram_total_freq, columns = ['Total count'])
	bgs = pd.DataFrame(bigrams, columns = ["Bigram"])
	speech = pd.DataFrame(speeches, columns = ["Speechids"])
	speaker = pd.DataFrame(speakers, columns = ["Speakers"])

	bigram_info = pd.DataFrame()
	bigram_info = pd.concat([bgs, speech, speaker, bg_num_speeches, bg_num_speakers, bg_total_freq], axis = 1)
	writer = pd.ExcelWriter("bigram_info.xlsx")
	bigram_info.to_excel(writer, 'Sheet1')
	writer.save()

# Computes the counts and tfidf scores for each party for further analysis in R
# Can limit based on the number of people in each party that use that bigram and how
# frequently that bigram is used
def counts_and_tfidf(Girondins, Montagnards, gir_docs, mont_docs, num_speeches, bigram_doc_freq):
	
	# Computes the tfidf scores within each group
	gir_tfidf = compute_tfidf(Girondins, num_speeches, bigram_doc_freq)
	mont_tfidf = compute_tfidf(Montagnards, num_speeches, bigram_doc_freq)

	store_to_pickle(gir_tfidf, "gir_tfidf.pickle")
	store_to_pickle(mont_tfidf, "mont_tfidf.pickle")

	# Stores the tf_idf vectors in Excel
	df_gir_tfidf = pd.DataFrame.from_dict(gir_tfidf, orient = "index")
	write_to_excel(df_gir_tfidf, 'gir_tfidf.xlsx')

	df_mont_tfidf = pd.DataFrame.from_dict(mont_tfidf, orient = "index")
	write_to_excel(df_mont_tfidf, 'mont_tfidf.xlsx')

	# Combines the tfidf vectors of both parties into one file
	df_tfidf_combined = pd.DataFrame([gir_tfidf, mont_tfidf])
	df_tfidf_combined = df_tfidf_combined.transpose()
	df_tfidf_combined.columns = ["Girondins", "Montagnards"]
	write_to_excel(df_tfidf_combined, 'combined_tfidf.xlsx')

	# Limits based on v, or the number of times that bigram appears, and gir or mont docs, the number of 
	# speakers in each group that use that bigram
	# Can change the name of these dataframes to illuminate what the restrictions are
	Girondins_restricted = {k:v for k,v in Girondins.items() if (v >= 10)} #and (len(gir_docs[k]) > 1)}
	Montagnards_restricted = {k:v for k,v in Montagnards.items() if (v >= 10)} #and (len(mont_docs[k]) > 1)}

	store_to_pickle(Girondins_restricted, "Girondins_restricted.pickle")
	store_to_pickle(Montagnards_restricted, "Montagnards_restricted.pickle")

	gir_tfidf = compute_tfidf(Girondins, num_speeches, bigram_doc_freq)
	mont_tfidf = compute_tfidf(Montagnards, num_speeches, bigram_doc_freq)

	# Stores the Girondins and Montagnards frequency vectors and tfidfs in the same document according to restrictions
	df_combined = pd.DataFrame([Girondins, Montagnards])
	df_combined = df_combined.transpose()
	df_combined.columns = ["Girondins", "Montagnards"]
	write_to_excel(df_combined, 'combined_frequency_restricted.xlsx')

	df_tfidf_combined = pd.DataFrame([gir_tfidf, mont_tfidf])
	df_tfidf_combined = df_tfidf_combined.transpose()
	df_tfidf_combined.columns = ["Girondins", "Montagnards"]
	write_to_excel(df_tfidf_combined, 'combined_tfidf_restricted.xlsx')


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

if __name__ == '__main__':
    import sys
    raw_speeches = pickle.load(open("raw_speeches.pickle", "rb"))
    speechid_to_speaker = pickle.load(open("speechid_to_speaker.pickle", "rb"))
    speakers_to_analyze = load_list("Girondins and Montagnards New Mod Limit.xlsx")
    Girondins = Counter()
    Montagnards = Counter()
    try:
    	os.mkdir('../Speakers')
    except OSError:
    	pass
    aggregate(speakers_to_analyze, raw_speeches, speechid_to_speaker, Girondins, Montagnards)
