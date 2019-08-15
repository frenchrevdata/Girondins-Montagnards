#!/usr/bin/env python
# -*- coding=utf-8 -*-

"""
Plots the CDFs of the Girondins and Montagnards predictions from the classification analysis
"""

import pickle
import pandas as pd
from pandas import *
import numpy as np
import collections
from collections import Counter
import matplotlib.pyplot as plt


def create_arrays(real_pred):
	gir = real_pred
	gir_cols_drop = []
	mont = real_pred
	mont_cols_drop = []

	# Limit the analysis to only the corrected predicted entries
	# Girondins is 0 and Montagnards is 1
	for i, index in enumerate(real_pred.index.values):
		if (real_pred['Real classification'].iloc[i] == 1) or (real_pred['Predicted'].iloc[i] == 1):
			gir_cols_drop.append(i)
		if (real_pred['Real classification'].iloc[i] == 0) or (real_pred['Predicted'].iloc[i] == 0):
			mont_cols_drop.append(i)
	gir = gir.drop(gir_cols_drop)
	mont = mont.drop(mont_cols_drop)
	return ([gir, mont])


def plot_cdf(gir_array, mont_array):
	gir_probs = gir_array['Prob 0']
	mont_probs = mont_array['Prob 1']

	num_bins = 50
	
	plt.hist(gir_probs, bins = num_bins, histtype = 'step', density = True, cumulative = True, label = 'Gir')
	plt.hist(mont_probs, bins = num_bins, histtype = 'step', density = True, cumulative = True, label = 'Mont')

	# This title will change according to which model is being analyzed
	plt.title("CDFs for Logistic Regression Model")

	plt.legend(loc = 'upper left')
	plt.show()
	plt.savefig('classification_values.png')


if __name__ == '__main__':
	import sys
	
	# Only one of the following lines will be used depending on which model is being considered
	real_pred = pickle.load(open("real_pred_lr.pickle", "rb"))
	#real_pred = pickle.load(open("real_pred_xgb.pickle", "rb"))

	gir_array, mont_array = create_arrays(real_pred)
	plot_cdf(gir_array, mont_array)
