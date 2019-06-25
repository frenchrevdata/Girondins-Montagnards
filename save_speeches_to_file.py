import pickle
import os

with open("raw_speeches.pickle", "rb") as f:
	raw_speeches = pickle.load(f)
try:
	os.mkdir('../Speeches')
except OSError:
	pass
for speechid in raw_speeches:
	pickle_filename = "../Speeches/" + speechid + ".pickle"
	with open(pickle_filename, 'wb') as handle:
		pickle.dump(raw_speeches[speechid], handle, protocol = 0)