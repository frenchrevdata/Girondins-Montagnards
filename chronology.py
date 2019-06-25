#!/usr/bin/env python
# -*- coding=utf-8 -*-

import pickle
import pandas as pd
from pandas import *
import numpy as np
import csv
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
from bokeh.plotting import figure

date_regex = '([0-9]{4}-[0-9]{2}-[0-9]{1,2})'

def calculate_chronology(raw_speeches, speechid_to_speaker, speaker_list, speakers_to_analyze, Girondins, Montagnards):
	speaker_ngrams = {}
	speakers_to_consider = []
	speaker_distances = collections.defaultdict()
	chronology = collections.defaultdict(dict)
	# chronology_date = pd.DataFrame(columns = ["Bigram", "Speaker Name", "Date", "Num occurrences"])
	# chronology_speechid = pd.DataFrame(columns = ["Bigram", "Speaker Name", "Speechid", "Num occurrences"])


	for speaker in speakers_to_analyze.index.values:
		speakers_to_consider.append(remove_diacritic(speaker).decode('utf-8'))

	row_entry_speechid = []
	row_entry_date = []
	for identity in raw_speeches:
		date = re.findall(date_regex, str(identity))[0]
		speaker_name = speechid_to_speaker[identity]
		# print speaker_name
		if (date >= "1792-09-20") and (date <= "1793-06-02") and (speaker_name in speakers_to_consider):
			party = speakers_to_analyze.loc[speaker_name, "Party"]
			indv_speech_bigram = compute_ngrams(raw_speeches[identity], 2)
			for bigram in indv_speech_bigram:
				row_entry_speechid.append([str(bigram), speaker_name, identity, indv_speech_bigram[bigram], party])
				# chronology_speechid = chronology_speechid.append(pd.Series(row_entry_speechid), ignore_index = True)
				row_entry_date.append([str(bigram), speaker_name, date, indv_speech_bigram[bigram], party])
				# chronology_date = chronology_date.append(pd.Series(row_entry_date), ignore_index = True)
				# if bigram in chronology:
				# 	chronology[bigram].append([speaker_name, identity, indv_speech_bigram[bigram]])
				# else:
				# 	chronology[bigram] = []
				# 	chronology[bigram].append([speaker_name, identity, indv_speech_bigram[bigram]])

	chronology_speechid = pd.DataFrame(row_entry_speechid, columns = ["Bigram", "Speaker Name", "Speechid", "Num occurrences", "Party"])
	chronology_date = pd.DataFrame(row_entry_date, columns = ["Bigram", "Speaker Name", "Date", "Num occurrences", "Party"])



	# Create ngram column, speaker name, date, number of occurrences
	# Create two dataframes, one with date and one with speechid
	# Include volume number
	# Do groupby and aggregation methods

	# w = csv.writer(open("chronology.csv", "w"))
	# for key, val in chronology.items():
	# 	if (Girondins[key] >= 10) or (Montagnards[key] >= 10):
	# 		w.writerow([key,val])
	make_visualizations(chronology_date)

	# write_to = pd.ExcelWriter("chronology_speechid.xlsx")
	# chronology_speechid.to_excel(write_to, 'Sheet1')
	# write_to.save()

	# filename = pd.ExcelWriter("chronology_date.xlsx")
	# chronology_date.to_excel(write_to, 'Sheet1')
	# filename.save()


	pickle_filename_2 = "chronology_speechid.pickle"
	with open(pickle_filename_2, 'wb') as handle:
		pickle.dump(chronology_speechid, handle, protocol = 0)

	pickle_filename = "chronology_date.pickle"
	with open(pickle_filename, 'wb') as handle:
		pickle.dump(chronology_date, handle, protocol = 0)

def make_visualizations(chronology_date):
	# chronology_date = pd.DataFrame.from_dict(pickle.load(open("chronology.pickle", "rb")), orient = "index")
	# chronology_date.columns = ["Bigram", "Speaker Name", "Date", "Num occurrences"]
	# print(chronology_date)
	# chronology_date["Num per date"] = chronology_date["Num occurrences"].groupby(chronology_date["Date"]).sum()
	# num_per_date = chronology_date.groupby(["Date"]).agg({"Num occurrences": "sum"})
	# print "Num per date"
	# print num_per_date
	num_per_bigram_per_date = chronology_date.groupby(["Bigram", "Date"]).agg({"Num occurrences": "sum"})
	# with open("num_per_bigram_date.pickle", "wb") as handle:
	# 	pickle.dump(num_per_bigram_per_date, handle, protocol = 2)
	# num_bigram_date = pickle.load(open("num_per_bigram_date.pickle","rb"))
	grouped = chronology_date.groupby(["Bigram"])
	with open("bybigram.pickle", "wb") as handle:
		pickle.dump(grouped, handle, protocol = 2)
	# for name, group in grouped:
	# 	print "\n"
	# 	print name
	# 	print "\n"
	# 	print group
	# print "num per bigram per date"
	# print num_per_bigram_per_date
	# num_per_bigram_per_speaker = chronology_date.groupby(["Bigram", "Speaker Name"]).agg({"Num occurrences": "sum"})
	# print "num per bigram per speaker"
	# print num_per_bigram_per_speaker
	# print(chronology_date)




if __name__ == '__main__':
	import sys
	raw_speeches = pickle.load(open("raw_speeches.pickle", "rb"))
	speechid_to_speaker = pickle.load(open("speechid_to_speaker.pickle", "rb"))
	Girondins = pickle.load(open("Girondins_withlimit.pickle", "rb"))
	Montagnards = pickle.load(open("Montagnards_withlimit.pickle", "rb"))

	# chronology = pickle.load(open("chronology.pickle", "rb"))
	speaker_list = load_speakerlist('Copy of AP_Speaker_Authority_List_Edited_3.xlsx')

	speakers_to_analyze = load_list("Girondins and Montagnards New Mod.xlsx")

	calculate_chronology(raw_speeches, speechid_to_speaker, speaker_list, speakers_to_analyze, Girondins, Montagnards)
	# make_visualizations(speechid_to_speaker)