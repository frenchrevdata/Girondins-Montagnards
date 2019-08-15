#!/usr/bin/env python
# -*- coding=utf-8 -*-

import nltk
import collections
from nltk import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
from collections import Counter
import pickle
import unicodedata

def compute_bigrams(input):
    token = word_tokenize(input)
    bigrams = ngrams(token, 2)
    return bigrams

def compute_trigrams(input):
    token = word_tokenize(input)
    trigrams = ngrams(token, 3)
    return trigrams

def compute_fourgrams(input):
    token = word_tokenize(input)
    fourgrams = ngrams(token, 4)
    return fourgrams

def remove_diacritic(input):
    '''
    Accept a unicode string, and return a normal string (bytes in Python 3)
    without any diacritical marks.
    '''
    return unicodedata.normalize('NFKD', input).encode('ASCII', 'ignore')

def load_stopwords(textfile):
    stopwords = open(textfile, 'r')
    lines = stopwords.readlines()
    french_stopwords = []
    for line in lines:
        word = line.split(',')
        #remove returns and new lines at the end of stop words so the parser catches matches
        #also remove accents so the entire analysis is done without accents
        word_to_append = remove_diacritic(unicode(word[0].replace("\n","").replace("\r",""), 'utf-8'))
        french_stopwords.append(word_to_append)
    return(french_stopwords)


def remove_stopwords(input, stopwordfile):
    """french_stopwords = stopwords.words('french')
    cleaned_stopwords = []
    cleaned_stopwords.append("les")
    for stopword in french_stopwords:
        cleaned_stopwords.append(remove_diacritic(stopword))"""
    filtered_text = ""
    """for word in input.split():
        if word not in french_stopwords:
            if word != "les" and word != "a":
                filtered_text = filtered_text + " " + word"""
    for word in input.split():
        if word not in stopwordfile:
            filtered_text = filtered_text + " " + word
    return filtered_text

if __name__ == '__main__':
    import sys
    
    filename = sys.argv[1]
    stopwords_file = sys.argv[2]

    stopwords = load_stopwords(stopwords_file)

    with open(filename, 'rb') as handle:
        dictionary = pickle.load(handle)

    """text = ""
    for entry in dictionary:
        text = text + dictionary[entry]
    text = text.replace(",", "").replace(".", "").replace(":","")
    clean_text = remove_stopwords(text)"""

    #remove all punctuation
    text = dictionary['Vernier'].replace("'"," ").replace(";"," ").replace(",", " ").replace(":"," ").replace("."," ").replace("("," ").replace(")"," ")
    #convert text to lowercase to align with stopword list
    clean_text = remove_stopwords(text.lower(), stopwords)

    new_bigrams = compute_bigrams(clean_text)
    bigram_count = Counter(new_bigrams)
    print("BIGRAMS")
    print(bigram_count.most_common(10))
    print("\n")

    new_trigrams = compute_trigrams(clean_text)
    trigram_count = Counter(new_trigrams)
    print("TRIGRAMS")
    print(trigram_count.most_common(10))
    print("\n")

    new_fourgrams = compute_fourgrams(clean_text)
    fourgram_count = Counter(new_fourgrams)
    print("FOURGRAMS")
    print(fourgram_count.most_common(10))
    print("\n")
