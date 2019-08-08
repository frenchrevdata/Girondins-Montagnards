#!/usr/bin/env python
# -*- coding=utf-8 -*-

""" Analyzes bigrams over time """

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
from processing_functions import load_list, load_speakerlist, process_excel, remove_diacritic, write_to_excel, store_to_pickle
from scipy import spatial
from bokeh.plotting import figure

date_regex = '([0-9]{4}-[0-9]{2}-[0-9]{1,2})'

def calculate_chronology(raw_speeches, speechid_to_speaker, speaker_list, speakers_to_analyze, Girondins, Montagnards):
	speaker_ngrams = {}
	speakers_to_consider = []
	speaker_distances = collections.defaultdict()
	chronology = collections.defaultdict(dict)

	for speaker in speakers_to_analyze.index.values:
		speakers_to_consider.append(remove_diacritic(speaker).decode('utf-8'))

	row_entry_speechid = []
	row_entry_date = []
	for identity in raw_speeches:
		date = re.findall(date_regex, str(identity))[0]
		speaker_name = speechid_to_speaker[identity]
		if (date >= "1792-09-20") and (date <= "1793-06-02") and (speaker_name in speakers_to_consider):
			party = speakers_to_analyze.loc[speaker_name, "Party"]
			indv_speech_bigram = compute_ngrams(raw_speeches[identity], 2)
			# Store relevant information for each bigram
			for bigram in indv_speech_bigram:
				row_entry_speechid.append([str(bigram), speaker_name, identity, indv_speech_bigram[bigram], party])
				row_entry_date.append([str(bigram), speaker_name, date, indv_speech_bigram[bigram], party])


	chronology_speechid = pd.DataFrame(row_entry_speechid, columns = ["Bigram", "Speaker Name", "Speechid", "Num occurrences", "Party"])
	chronology_date = pd.DataFrame(row_entry_date, columns = ["Bigram", "Speaker Name", "Date", "Num occurrences", "Party"])



	# w = csv.writer(open("chronology.csv", "w"))
	# for key, val in chronology.items():
	# 	if (Girondins[key] >= 10) or (Montagnards[key] >= 10):
	# 		w.writerow([key,val])
	make_visualizations(chronology_date)

	write_to_excel(chronology_speechid, "chronology_speechid.xlsx")
	write_to_excel(chronology_date, "chronology_date.xlsx")

	store_to_pickle(chronology_speechid, "chronology_speechid.pickle")
	store_to_pickle(chronology_date, "chronology_date.pickle")


# Aggreate chronology_date for purposes of visualizations
def make_visualizations(chronology_date):
	
	num_per_bigram_per_date = chronology_date.groupby(["Bigram", "Date"]).agg({"Num occurrences": "sum"})
	store_to_pickle(num_per_bigram_per_date, "num_per_bigram_per_date.pickle")

	# num_bigram_date = pickle.load(open("num_per_bigram_date.pickle","rb"))
	grouped = chronology_date.groupby(["Bigram"])
	store_to_pickle(grouped, "grouped.pickle")




if __name__ == '__main__':
	import sys
	raw_speeches = pickle.load(open("raw_speeches.pickle", "rb"))
	speechid_to_speaker = pickle.load(open("speechid_to_speaker.pickle", "rb"))
	Girondins = pickle.load(open("Girondins.pickle", "rb"))
	Montagnards = pickle.load(open("Montagnards.pickle", "rb"))

	# chronology = pickle.load(open("chronology.pickle", "rb"))
	speaker_list = load_speakerlist('Copy of AP_Speaker_Authority_List_Edited_3.xlsx')

	speakers_to_analyze = load_list("Girondins and Montagnards New Mod.xlsx")

	calculate_chronology(raw_speeches, speechid_to_speaker, speaker_list, speakers_to_analyze, Girondins, Montagnards)
