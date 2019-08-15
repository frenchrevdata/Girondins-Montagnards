#!/usr/bin/env python
# -*- coding=utf-8 -*-

from bs4 import BeautifulSoup
import unicodedata
import os
import csv
import pickle
import re

def remove_diacritic(input):
    '''
    Accept a unicode string, and return a normal string (bytes in Python 3)
    without any diacritical marks.
    '''
    return unicodedata.normalize('NFKD', input).encode('ASCII', 'ignore')

def parseFile(file):
    file = open(sys.argv[1], "r")
    contents = file.read()
    soup = BeautifulSoup(contents, 'lxml')
    
    doc_dates = extractDates(soup)

    speaker_text_dict = {}

    for talk in soup.find_all('sp'):
        # Key is speaker name
        key = talk.find('speaker').get_text()
        key = remove_diacritic(key).decode('utf-8')
        if key.endswith('.'):
            key = key[:-1]
        # Value is speaker speech
        value = talk.find('p').get_text()
        value = remove_diacritic(value).decode('utf-8')

        speech = talk.find_all('p')
        text = ""
        for section in speech:
            text = text + section.get_text()
        value = remove_diacritic(text).decode('utf-8')

        if key and value:
            # Check if the key already exists
            if speaker_text_dict.has_key(key):
                current_value = speaker_text_dict[key]
                new_value = current_value + " " + value
                new_value = new_value.replace("\n", " ")
                new_value = re.sub(r'([ ]{2,})', ' ', new_value)
                speaker_text_dict[key] = new_value
            else:
                value = value.replace("\n", " ")
                value = re.sub(r'([ ]{2,})', ' ', value)
                speaker_text_dict[key] = value

    csv_filename = "" + doc_dates[0] + " a " + doc_dates[1] + ".csv"
    pickle_filename = "" + doc_dates[0] + " a " + doc_dates[1] + ".pickle"
    # Writes the dictionary to a csv file to sanity check. Still some parsing errors
    with open(pickle_filename, 'wb') as handle:
        pickle.dump(speaker_text_dict, handle, protocol = 0)



    """with open(csv_filename, 'wb') as csvfile:
        writer = csv.writer(csvfile)
        for key, value in speaker_text_dict.items():
            writer.writerow([key, value])"""

    file.close()

    #print(speaker_text_dict.keys())

# Parses dates from file being analyzed
def extractDates(soup_file):
    dates = soup_file.find_all('date')
    relevant_dates = []
    for date in dates:
        if date.attrs:
            relevant_dates.append(date)
    return([relevant_dates[0]['value'], relevant_dates[1]['value']])


# Code for parsing all files but getting memory errors
"""def parseFiles():
    #file = open(sys.argv[1], "r")
    speaker_text_dict = {}
    for filename in os.listdir("Docs/"):
        if filename.endswith(".xml"):
            filename = open('Docs/' + filename, "r")
            contents = filename.read()
            soup = BeautifulSoup(contents, 'lxml')
            for talk in soup.find_all('sp'):
                # Key is speaker name
                if talk.find('speaker'):
                    key = talk.find('speaker').get_text()
                #key = talk.find('speaker').get_text()
                    key = remove_diacritic(key).decode('utf-8')
                
                # Value is speaker speech

                # Code for only one paragraph, mostly used for testing
                value = talk.find('p').get_text()
                value = remove_diacritic(value).decode('utf-8')

                # Code for when there are multiple paragraphs
                speech = talk.find_all('p')
                text = ""
                for section in speech:
                    text = text + section.get_text()
                value = remove_diacritic(text).decode('utf-8')

                # Make sure that key and value exist
                if key and value:
                    # Check if the key already exists
                    if speaker_text_dict.has_key(key):
                        current_value = speaker_text_dict[key]
                        new_value = current_value + value
                        speaker_text_dict[key] = new_value
                    else:
                        speaker_text_dict[key] = value
            filename.close()
        else:
            continue
    print(speaker_text_dict['M. Boutteville- Dumetz.'])
    #print(speaker_text_dict.keys())"""


if __name__ == '__main__':
    import sys
    
    
    #parseFiles()
    
    file_name = sys.argv[1]
    parseFile(file_name)
