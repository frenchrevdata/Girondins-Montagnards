#!/usr/bin/env python
# -*- coding=utf-8 -*-

"""
Computes n-grams of a specified length on a given text
"""

# import pandas as pd
import numpy as np
from nltk import word_tokenize
from nltk.util import ngrams
import collections
from collections import Counter
from processing_functions import remove_diacritic

# Tokenize the input text according to the given amount with '1' being unigram, '2' being bigram, etc.
def make_ngrams(input, amount):
	token = word_tokenize(input)
	n_grams = ngrams(token, amount)
	return n_grams

# Removes a set of predefined stop words
def remove_stopwords(input, stopwords):
	filtered_text = ""
	for word in input.split():
		if word not in stopwords:
			filtered_text = filtered_text + " " + word
	return filtered_text

# Cleans the text and then calls make_ngrams to develop the n-grams on the cleaned text
def compute_ngrams(speech, order):
	stopwords_from_file = open('FrenchStopwords.txt', 'r')
	lines = stopwords_from_file.readlines()
	french_stopwords = []
	for line in lines:
		word = line.split(',')
		#remove returns and new lines at the end of stop words so the parser catches matches
		#also remove accents so the entire analysis is done without accents
		word_to_append = remove_diacritic(unicode(word[0].replace("\n","").replace("\r",""), 'utf-8'))
		french_stopwords.append(word_to_append)

	speech = speech.replace("%"," ").replace("\\"," ").replace("^", " ").replace("=", " ").replace("]"," ").replace("\""," ").replace("``", " ").replace("-"," ").replace("[", " ").replace("{"," ").replace("$", " ").replace("~"," ").replace("-"," ").replace("}", " ").replace("&"," ").replace(">"," ").replace("#"," ").replace("/"," ").replace("\`"," ").replace("'"," ").replace("*", " ").replace("`", " ").replace(";"," ").replace("?"," ").replace(",", " ").replace(":"," ").replace("."," ").replace("("," ").replace(")"," ")
	clean_text = remove_stopwords(speech.lower(), french_stopwords)
	clean_text = clean_text.replace("mm secretaire", " ").replace("assemble nationale", " ").replace("monsieur president", " ").replace("convention nationale", " ").replace("archives parliamentaire", " ").replace("republique francaise", " ").replace("ordre jour", " ").replace("corps legislatif", " ")
	n_grams = make_ngrams(clean_text, order)
	speech_ngrams = Counter(n_grams)
	return(speech_ngrams)