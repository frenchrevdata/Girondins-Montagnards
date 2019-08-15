#!/usr/bin/env python
# -*- coding=utf-8 -*-

from bs4 import BeautifulSoup
import unicodedata
import os
import csv
import pickle
import regex as re
import pandas as pd
import numpy as np
from nltk import word_tokenize
from nltk.util import ngrams
import collections
from collections import Counter
import os
import gzip



#daily_regex = '(?:<p>[ ]{0,1}Séance)(?:[\s\S]+)(?:<p>[ ]{0,1}Séance)'
#Seance followed by less than or equal to 4 line breaks (\n) then date value =
daily_regex = '(?:Séance[\s\S]{0,200}<date value=\")(?:[\s\S]+)(?:Séance[\s\S]{0,200}<date value=\")'
speechid_to_speaker = {}
speeches_per_day = {}
dates = set()
names_not_caught = set()
global stopwords
global speaker_list



def remove_diacritic(input):
    '''
    Accept a unicode string, and return a normal string (bytes in Python 3)
    without any diacritical marks.
    '''
    return unicodedata.normalize('NFKD', input).encode('ASCII', 'ignore')


def parseFiles():
    files = os.listdir("Docs/")
    for filename in files:
        if filename.endswith(".xml"):
        	print(filename)
        	filename = open('Docs/' + filename, "r")
        	contents = filename.read()
        	soup = BeautifulSoup(contents, 'lxml')
        	volDates = extractVolDates(soup)
        	sessions = soup.find_all(['div2', 'div3'], {"type": ["session", "other"]})
        	#sessions.append(soup.find_all('div2', {"type": "other"}))
        	for session in sessions:
		        date = extractDate(session)
		        if (date >= "1789-05-05") and (date <= "1795-01-04") and (date != "error"):
		        	if date in dates:
		        		date = date + "_soir"
		        		if date in dates:
		        			date = date + "2"
		        			findSpeeches(session, date)
		        		else:
		        			findSpeeches(session, date)
		        			dates.add(date)		        		
		        	else:
		        		findSpeeches(session, date)
		        		dates.add(date)
	        filename.close()


def findSpeeches(daily_soup, date):
	id_base = date.replace("/","_")
	number_of_speeches = 0
	speeches_of_day = ""
	dict_of_speeches = {}
	for talk in daily_soup.find_all('sp'):
		try:
			speaker = talk.find('speaker').get_text()
			speaker = remove_diacritic(speaker).decode('utf-8')
			speaker = speaker.replace("-"," ")
			if speaker.endswith('.'):
				speaker = speaker[:-1]
			if speaker.endswith(","):
				speaker = speaker[:-1]
			if speaker.endswith(", "):
				speaker = speaker[:-1]
			if speaker.startswith('M.'):
				speaker = speaker[2:]
			if speaker.startswith('M '):
				speaker = speaker[2:]
			if speaker.startswith('MM. '):
				speaker = speaker[4:]
			if speaker.startswith(' M. '):
				speaker = speaker[3:]
			if speaker.startswith(' '):
				speaker = speaker[1:]
			if speaker.endswith(' '):
				speaker = speaker[:-1]
			if speaker.endswith('.'):
				speaker = speaker[:-1]
		except AttributeError:
			speaker = ""
		speech = talk.find_all('p')
		text = ""
		full_speech = ""
		for section in speech:
			text = text + section.get_text()
		full_speech = remove_diacritic(text).decode('utf-8')
		full_speech = full_speech.replace("\n"," ").replace("--"," ").replace("!"," ")
		full_speech = re.sub(r'([ ]{2,})', ' ', full_speech)
		full_speech = re.sub(r'([0-9]{1,4})', ' ', full_speech)
		speaker_name = ""
		#speech_of_day = speech_of_day + full_speech
		if speaker != "Le President":
			if speaker in speaker_list.index.values:
				#number_of_speeches = number_of_speeches + 1
				#speeches_of_day = speeches_of_day + " " + full_speech
				speaker_name = speaker_list.loc[speaker, "FullName"]
				"""speech_id = "" + id_base + "_" + str(number_of_speeches)
				speechid_to_speaker[speech_id] = speaker_name
				dict_of_speeches[speech_id] = full_speech"""
			else:
				for i, name in enumerate(speaker_list['LastName']):
					if speaker.find(name) != -1:
						speaker_name = speaker_list["FullName"].iloc[i]
		if speaker_name is not "":
			number_of_speeches = number_of_speeches + 1
			speeches_of_day = speeches_of_day + " " + full_speech
			speech_id = "" + id_base + "_" + str(number_of_speeches)
			speechid_to_speaker[speech_id] = speaker_name
			dict_of_speeches[speech_id] = full_speech
		else:
			names_not_caught.add(speaker)
	
	# Computes ngrams on all the speeches from the given day
	compute_ngrams(id_base, speeches_of_day)

	# Serializes the dictionary to a pickle file to sanity check.
	try:
		os.mkdir('../Speeches')
	except OSError:
		pass
	pickle_filename = "../Speeches/" + id_base + "_speeches.pickle"
	with open(pickle_filename, 'wb') as handle:
		pickle.dump(dict_of_speeches, handle, protocol = 0)
	

	speeches_per_day[id_base] = number_of_speeches


def make_ngrams(input, amount):
	token = word_tokenize(input)
	n_grams = ngrams(token, amount)
	return n_grams


def load_stopwords(textfile):
	stopwords_from_file = open(textfile, 'r')
	lines = stopwords_from_file.readlines()
	french_stopwords = []
	for line in lines:
		word = line.split(',')
		#remove returns and new lines at the end of stop words so the parser catches matches
		#also remove accents so the entire analysis is done without accents
		word_to_append = remove_diacritic(unicode(word[0].replace("\n","").replace("\r",""), 'utf-8'))
		french_stopwords.append(word_to_append)
	return(french_stopwords)


def load_speakerlist(speakernames):
	pd_list = pd.read_excel(speakernames, sheet_name= 'AP Speaker Authority List xlsx')
	pd_list = pd_list.set_index('Names')
	speakers = pd_list.index.tolist()
	for speaker in speakers:
		ind = speakers.index(speaker)
		speakers[ind] = remove_diacritic(speaker).decode('utf-8')
	pd_list.index = speakers
	return pd_list

 	


def remove_stopwords(input):
	filtered_text = ""
	for word in input.split():
		if word not in stopwords:
			filtered_text = filtered_text + " " + word
	return filtered_text


def compute_ngrams(uniqueid, speech):
	speech = speech.replace("'"," ").replace("*", " ").replace("`", " ").replace(";"," ").replace(",", " ").replace(":"," ").replace("."," ").replace("("," ").replace(")"," ")
	clean_text = remove_stopwords(speech.lower())
	clean_text = clean_text.replace("mm secretaire", " ").replace("assemble nationale", " ").replace("monsieur president", " ").replace("convention nationale", " ").replace("archives parliamentaire", " ").replace("republique francaise", " ").replace("ordre jour", " ").replace("corps legislatif", " ")
	n_grams = make_ngrams(clean_text, 2)
	speech_ngrams = Counter(n_grams)
	try:
		os.mkdir('../Ngrams')
	except OSError:
		pass
	#txt_filename = "../Ngrams/" + uniqueid + "_ngrams" + ".txt.gz"
	txt_filename = "../Ngrams/" + uniqueid + "_ngrams" + ".txt"
	# Serializes the dictionary to a pickle file to sanity check. 
	#txtfile = gzip.open(txt_filename, 'wb')
	txtfile = open(txt_filename, 'w')
	txtfile.write(str(speech_ngrams))
	txtfile.close()
	"""pickle_filename = "" + uniqueid + "_ngrams" + ".pickle"
	# Serializes the dictionary to a pickle file to sanity check. 
	with open(pickle_filename, 'wb') as handle:
		pickle.dump(speech_ngrams, handle, protocol = 0)"""
   


# Parses dates from file being analyzed
def extractDate(soup_file):
	dates = soup_file.find_all('date')
	relevant_dates = []
	for date in dates:
		if date.attrs:
			relevant_dates.append(date)
	if (len(relevant_dates) > 0):
		return(relevant_dates[0]['value'])
	else:
		return("error")

def extractVolDates(soup_file):
    dates = soup_file.find_all('date')
    relevant_dates = []
    for date in dates:
        if date.attrs:
            relevant_dates.append(date)
    return([relevant_dates[0]['value'], relevant_dates[1]['value']])




if __name__ == '__main__':
    import sys
    stopwords = load_stopwords('FrenchStopwords.txt')
    speaker_list = load_speakerlist('AP_Speaker_Authority_List_Edited_2.xlsx')
    parseFiles()
    txt_filename = "" + "Names_not_caught" + ".txt"
    txtfile = open(txt_filename, 'w')
    txtfile.write(str(names_not_caught))
    txtfile.close()


    
       
   	
