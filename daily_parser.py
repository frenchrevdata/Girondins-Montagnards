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

#daily_regex = '(?:<p>[ ]{0,1}Séance)[\s\S]+(?:<p>[ ]{0,1}Séance)'
daily_regex = '(?:Séance[\s\S]{0,200}<date value=\")(?:[\s\S]+)(?:Séance[\s\S]{0,200}<date value=\")'


def remove_diacritic(input):
    '''
    Accept a unicode string, and return a normal string (bytes in Python 3)
    without any diacritical marks.
    '''
    return unicodedata.normalize('NFKD', input).encode('ASCII', 'ignore')

#NEED TO ADD CODE TO PARSE INDIVIDUAL SPEECHES NOT JUST SPEAKER ON A GIVEN DAY

def parseFile(file):
    file = open(sys.argv[1], "r")
    contents = file.read()

    soup = BeautifulSoup(contents, 'lxml')
    sessions = soup.find_all('div2', {"type": "session"})
    
    #daily = re.findall(daily_regex, contents, overlapped=True)
    #for day in daily:
    for session in sessions:
        #session_soup = BeautifulSoup(session, 'lxml')
        #date = extractDate(soup)
        date = extractDate(session)

        speaker_text_dict = {}
        for talk in session.find_all('sp'):
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
                    #new_value = re.sub(r'([ ]{2,})', ' ', new_value)
                    speaker_text_dict[key] = new_value
                else:
                    value = value.replace("\n", " ")
                    #value = re.sub(r'([ ]{2,})', ' ', value)
                    speaker_text_dict[key] = value

        # Using pandas to create a dataframe
        speaker_dataframe = pd.DataFrame.from_dict(speaker_text_dict, orient='index')


        """pickle_filename = "" + date + ".pickle"
        # Serializes the dictionary to a pickle file to sanity check. 
        with open(pickle_filename, 'wb') as handle:
            pickle.dump(speaker_text_dict, handle, protocol = 0)"""
   
    file.close()


# Parses dates from file being analyzed
def extractDate(soup_file):
    dates = soup_file.find_all('date')
    relevant_dates = []
    for date in dates:
        if date.attrs:
            relevant_dates.append(date)
    print(relevant_dates[0]['value'])
    return(relevant_dates[0]['value'])



if __name__ == '__main__':
    import sys
    
       
    file_name = sys.argv[1]
    parseFile(file_name)
