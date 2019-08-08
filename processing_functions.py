#!/usr/bin/env python
# -*- coding=utf-8 -*-

"""
This file is a collection of functions used multiple times throughout other scripts.
It avoids a repetition of data.
"""

import pickle
import unicodedata
import pandas as pd
from pandas import *
import numpy as np
import math
import csv


# Removes accents from the input parameter
def remove_diacritic(input):
    return unicodedata.normalize('NFKD', input).encode('ASCII', 'ignore')

"""
def print_to_excel(dict1, dict2, filename):
	df = pd.DataFrame([dict1, dict2])
	df = df.transpose()
	df.columns = ["Girondins", "Montagnards"]
	writer = pd.ExcelWriter(filename)
	df.to_excel(writer, 'Sheet1')
	writer.save()
"""

# Writes the given dataframe to the provided filename
# Assumes that the df parameter is a pandas dataframe
def write_to_excel(df, filename):
	writer = pd.ExcelWriter(filename)
	df.to_excel(writer, 'Sheet1')
	writer.save()


def write_to_csv(df, filename):
	w = csv.writer(open(filename, "w"))
	for key, val in df.items():
		w.writerow([key,val])

# Stores a pickle file of the passed in dataframe
def store_to_pickle(file_to_store, filename):
	with open(filename, 'wb') as handle:
		pickle.dump(file_to_store, handle, protocol = 0)

# Loads an unspecified list of speakers into a pandas list from Excel
def load_list(speakernames):
	pd_list = pd.read_excel(speakernames, sheet_name = 'Sheet1')
	pd_list = pd_list.set_index('Name')
	speakers = pd_list.index.tolist()
	for speaker in speakers:
		ind = speakers.index(speaker)
		speakers[ind] = remove_diacritic(speaker).decode('utf-8')
	pd_list.index = speakers
	return pd_list

# Loads the comprehensive speaker list into a pandas dataframe from Excel
def load_speakerlist(speakernames):
	pd_list = pd.read_excel(speakernames, sheet_name= 'AP Speaker Authority List xlsx')
	pd_list = pd_list.set_index('Names')
	speakers = pd_list.index.tolist()
	for speaker in speakers:
		ind = speakers.index(speaker)
		speakers[ind] = remove_diacritic(speaker).decode('utf-8')
	pd_list.index = speakers
	return pd_list

# Imports a given Excel file for use in the Girondins/Montagnards analysis
# Make sure that the header of the first column in "Bigrams"
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

# Computes the tf-idf score for given parameters
def compute_tfidf(dictionary, num_speeches, doc_freq):
	tfidf = {}
	for ngram in dictionary:
		if ngram in doc_freq:
			df = doc_freq[ngram]
		else:
			df = 1
		idf = math.log10(num_speeches) - math.log10(df)
		tf = dictionary[ngram]
		tfidf[ngram] = (1+math.log10(tf))*idf
	return tfidf

# Normalizes dictionaries for use in computing the distance between the two vectors
# Returns an array of the two normalized dictionaries
def normalize_dicts(first_dict, second_dict):
	dict1 = first_dict
	dict2 = second_dict
	all_sum = 0
	all_sum = all_sum + sum(dict1.values()) + sum(dict2.values())
	
	for key in dict1:
		dict1[key] = float(dict1[key])/all_sum

	for key in dict2:
		dict2[key] = float(dict2[key])/all_sum

	return([dict1, dict2])

# Need this function because the keys for one of the dictionaries were not strings so the match check
# in cosine similarity was not working
def convert_keys_to_string(dictionary):
	to_return = {}
	for k in dictionary:
		to_return[str(k)] = dictionary[k]
	return to_return

# Computes the difference between the values of two dictionaries
def compute_difference_withplein(dict1, dict2):
	diff = {}
	keys_seen = []
	for k in dict1:
		if k in dict2:
			keys_seen.append(k)
			diff[k] = dict1[k] - dict2[k]
		else:
			diff[k] = dict1[k]
	for key in dict2:
		if key not in keys_seen:
			diff[key] = dict2[key]
	return diff 

def compute_difference(dict1, dict2):
	diff = {}
	for k in dict1:
		diff[k] = dict1[k] - dict2[k]
	return diff

def normalize_by_speeches(dictionary, num_speeches):
	to_return = {}
	for k in dictionary:
		to_return[k] = dictionary[k]/(1.0*num_speeches)
	return to_return


# Computes the cosine similiarity between two dictionaries
# Cosine similarity is dot product over the norms of the two vectors
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