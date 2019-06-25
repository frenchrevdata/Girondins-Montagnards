#!/usr/bin/env python
# -*- coding=utf-8 -*-

import pickle
import csv
from processing_functions import load_list, remove_diacritic


def track_murmures_applaudissements(raw_speeches, speechid_to_speaker):
	speakers_to_analyze = load_list("Girondins and Montagnards New Mod Limit.xlsx")
	speakers_to_consider = []
	for speaker in speakers_to_analyze.index.values:
		speakers_to_consider.append(remove_diacritic(speaker).decode('utf-8'))
	murmures = []
	applaudissements = []
	Girondins_murmures = 0
	Montagnards_murmures = 0
	Girondins_applaudissements = 0
	Montagnards_applaudissements = 0
	murmures_by_date = {}
	applaudissements_by_date = {}
	total_murmures = 0
	total_applaudissements = 0
	murmures_speakers = {}
	applaudissements_speakers = {}
	for speechid, speech in raw_speeches.items():
		speaker_name = speechid_to_speaker[speechid]
		if speaker_name in speakers_to_consider:
			date = speechid[0:9]
			party = speakers_to_analyze.loc[speaker_name, "Party"]
			if "murmure" in speech:
				total_murmures += 1
				murmures.append(speechid)
				if party == "Girondins":
					Girondins_murmures += 1
				else:
					Montagnards_murmures += 1
				if date in murmures_by_date:
					murmures_by_date[date] += 1
				else:
					murmures_by_date[date] = 0
				if speaker_name in murmures_speakers:
					murmures_speakers[speaker_name] += 1
				else:
					murmures_speakers[speaker_name] = 0
			if "applaudissement" in speech:
				total_applaudissements += 1
				applaudissements.append(speechid)
				if party == "Girondins":
					Girondins_applaudissements += 1
				else:
					Montagnards_applaudissements += 1
				if date in applaudissements_by_date:
					applaudissements_by_date[date] += 1
				else:
					applaudissements_by_date[date] = 0
				if speaker_name in applaudissements_speakers:
					applaudissements_speakers[speaker_name] += 1
				else:
					applaudissements_speakers[speaker_name] = 0
		else:
			if "murmure" in speech:
				total_murmures += 1
			if "applaudissement" in speech:
				total_applaudissements += 1

	with open('gir_murmures.txt', 'w') as f:
		f.write('%d' % Girondins_murmures)
	with open('mont_murmures.txt', 'w') as f:
		f.write('%d' % Montagnards_murmures)
	print Montagnards_murmures + Girondins_murmures

	with open('total_murmures.txt', 'w') as f:
		f.write('%d' % total_murmures)
	with open('total_applaudissements.txt', 'w') as f:
		f.write('%d' % total_applaudissements)

	with open('gir_applaudissements.txt', 'w') as f:
		f.write('%d' % Girondins_applaudissements)
	with open('mont_applaudissements.txt', 'w') as f:
		f.write('%d' % Montagnards_applaudissements)
	print Montagnards_applaudissements + Girondins_applaudissements

	with open('murmures_by_date.pickle', 'wb') as handle:
		pickle.dump(murmures_by_date, handle, protocol = 0)

	with open('applaudissements_by_date.pickle', 'wb') as handle:
		pickle.dump(applaudissements_by_date, handle, protocol = 0)

	w = csv.writer(open("murmures_by_date.csv", "w"))
	for key, val in murmures_by_date.items():
		w.writerow([key, val])

	w = csv.writer(open("applaudissements_by_date.csv", "w"))
	for key, val in applaudissements_by_date.items():
		w.writerow([key, val])

	w = csv.writer(open("murmures_speakers.csv", "w"))
	for key, val in murmures_speakers.items():
		w.writerow([key, val])

	w = csv.writer(open("applaudissements_speakers.csv", "w"))
	for key, val in applaudissements_speakers.items():
		w.writerow([key, val])	

if __name__ == '__main__':
    import sys
    raw_speeches = pickle.load(open("raw_speeches.pickle", "rb"))
    speechid_to_speaker = pickle.load(open("speechid_to_speaker.pickle", "rb"))    

    track_murmures_applaudissements(raw_speeches, speechid_to_speaker)