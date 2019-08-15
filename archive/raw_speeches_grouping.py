#!/usr/bin/env python

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


def firststep():
	year_month = []
	full_date = []
	speaker = []
	ngrams = {}

	byyearmonth = pd.DataFrame()
	bydate = pd.DataFrame()
	byspeaker = pd.DataFrame()

	raw_speeches = pickle.load(open("raw_speeches.pickle", "rb"))
	dataframe = pd.DataFrame.from_dict(raw_speeches, orient = "index")
	dataframe.columns = ['Speeches']
	speechid_to_speaker = pickle.load(open("speechid_to_speaker.pickle", "rb"))
	file = open('num_speeches.txt', 'r')
	num_speeches = int(file.read())
	doc_freq = pickle.load(open("bigram_doc_freq.pickle", "rb"))

	for speechid in raw_speeches:
		speech_bigrams = compute_ngrams(raw_speeches[speechid], 2)
		ngrams[speechid] = speech_bigrams

		yearmonth = speechid[0:7]
		year_month.append(yearmonth)

		fulldate = speechid[0:10]
		full_date.append(fulldate)

		speaker.append(speechid_to_speaker[speechid])

	
	dataframe['Year-Month'] = pd.Series(year_month).values
	dataframe['Full Date'] = pd.Series(full_date).values
	dataframe['Speaker'] = pd.Series(speaker).values
	dataframe['Speechid'] = dataframe.index

	write_to_excel(dataframe, "raw_data.xlsx")
	"""with open("ngrams.pickle", "wb") as handle:
		pickle.dump(ngrams, handle, protocol = 0)"""

	"""byyearmonth['YearMonth'] = pd.Series(year_month).values
	byyearmonth['ngrams'] = pd.Series(ngrams).values

	byyearmonth_dict = pd.Series(byyearmonth.ngrams.values, index = byyearmonth.YearMonth).to_dict()

	with open("byyearmonth_dict.pickle", 'wb') as handle:
		pickle.dump(byyearmonth_dict, handle, protocol = 0)

	
	bydate['FullDate'] = pd.Series(full_date).values
	bydate['ngrams'] = pd.Series(ngrams).values

	bydate_dict = pd.Series(bydate.ngrams.values, index = bydate.FullDate).to_dict()

	with open("bydate_dict.pickle", 'wb') as handle:
		pickle.dump(bydate_dict, handle, protocol = 0)

	
	byspeaker['Speaker'] = pd.Series(speaker).values
	byspeaker['ngrams'] = pd.Series(ngrams).values

	byspeaker_dict = pd.Series(byspeaker.ngrams.values, index = byspeaker.Speaker).to_dict()

	with open("byspeaker_dict.pickle", 'wb') as handle:
		pickle.dump(byspeaker_dict, handle, protocol = 0)"""

	# compute ngrams for each speech
	# don't need tfidf because should just add the frequency vectors not the tfidf ones
	# extract year-month
	# extract year-month-date
	# make all of those individual columns and create a pandas dataframe
	# create a function for each grouping and do a pandas groupby

	"""byyearmonth = groupby_yearmonth(dataframe)
	write_to_excel(byyearmonth, "byyearmonth.xlsx")
	byyearmonth = None
	byspeaker = groupby_speaker(dataframe)
	write_to_excel(byspeaker, "byspeaker.xlsx")
	byspeaker = None
	bydate = groupby_date(dataframe)
	write_to_excel(bydate, "bydate.xlsx")
	bydate = None"""

	groupby_yearmonth(dataframe, ngrams)
	groupby_date(dataframe, ngrams)
	groupby_speaker(dataframe, ngrams)

def groupby_yearmonth(df, ngrams):
	byyearmonth_dict = {}

	for i, speechid in enumerate(df['Speechid']):
		yearmonth = df['Year-Month'].iloc[i]
		dict_ngrams = ngrams[speechid]
		if yearmonth in byyearmonth_dict:
			byyearmonth_dict[yearmonth] = byyearmonth_dict[yearmonth] + dict_ngrams
		else:
			byyearmonth_dict[yearmonth] = dict_ngrams

	byyearmonth = pd.DataFrame.from_dict(byyearmonth_dict, orient = "index")

	write_to_excel(byyearmonth, "byyearmonth.xlsx")
	with open("byyearmonth.pickle", "wb") as handle:
		pickle.dump(byyearmonth, handle, protocol = 0)


def groupby_speaker(df, ngrams):
	byspeaker_dict = {}

	for i, speechid in enumerate(df['Speechid']):
		speaker = df['Speaker'].iloc[i]
		dict_ngrams = ngrams[speechid]
		if speaker in byspeaker_dict:
			byspeaker_dict[speaker] = byspeaker_dict[speaker] + dict_ngrams
		else:
			byspeaker_dict[speaker] = dict_ngrams

	byspeaker = pd.DataFrame.from_dict(byspeaker_dict, orient = "index")

	write_to_excel(byspeaker, "byspeaker.xlsx")
	with open("byspeaker.pickle", "wb") as handle:
		pickle.dump(byspeaker, handle, protocol = 0)

def groupby_date(df, ngrams):
	bydate_dict = {}

	for i, speechid in enumerate(df['Speechid']):
		date = df['Full Date'].iloc[i]
		dict_ngrams = ngrams[speechid]
		if date in bydate_dict:
			bydate_dict[date] = byydate_dict[date] + dict_ngrams
		else:
			bydate_dict[date] = dict_ngrams

	bydate = pd.DataFrame.from_dict(bydate_dict, orient = "index")

	write_to_excel(bydate, "bydate.xlsx")
	with open("bydate.pickle", "wb") as handle:
		pickle.dump(bydate, handle, protocol = 0)


if __name__ == '__main__':
	import sys
	speaker_list = load_speakerlist('Copy of AP_Speaker_Authority_List_Edited_3.xlsx')
	firststep()
