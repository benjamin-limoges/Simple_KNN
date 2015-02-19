# TODO : implement ggplot instead of matplotlib

from sklearn import neighbors
from sklearn.datasets import load_iris
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import csv
import sys


# Global Variables
INFILE = "iris.csv"
MAX_NEIGHBORS = 9
PARTS = 5
SEED = 1878

def main( args ):

	# Main control for the file
	df, tup = setupkNN()

	kNN(df, tup, "Unnormalized")
	
	df = z_score_normalize(df, tup[0])
	
	kNN(df, tup, "Normalized")

# Returns a tuple with the dataframe, features list, label name,
# number of partitions (as an integer), and an array that labels
# the partition number for each data point.
def setupkNN():

	df, features = loadData()
	label = features.pop()

	# Assign fold number
	test_idx = assignRandom( len(df) )

	return df, (features, label, test_idx)

# Returns an array that has the permutation number for each data
def assignRandom( length ):

	np.random.seed( seed = SEED )
	test_idx = np.random.randint( 0, PARTS, length )

	return test_idx


# Tuple returned has a panadas dataframe in the 0th position
# In the first is the header of the csv
def loadData():

	np.random.seed(seed = SEED)

	df = pd.read_csv( INFILE )

	# Gets the names of the features, label
	infile_1 = open( INFILE )
	header = infile_1.next()
	infile_1.close()

	header = header.rstrip('\n')
	header = header.split(',')

	return df, header


# Implements the kth nearest neighbor algorithm and stores the results.
# Calls plots to produce graphs 
def kNN( df, tup, normalization ):
	
	features, label, test_idx = tup

	label = label.rstrip()

	results = {}
	w = ['uniform', 'distance']
	names = ['uniform', 'distance']

	for item in range(0, len(names)):
		result = []
		results[names[item]] = result

	# Iterates through the number of neighbors permitted.  Odd values
	# only.  To get the user specified number of neighbors, 1 is added
	# so that number is encompassed in the range.
	for n in range(1, MAX_NEIGHBORS + 1, 2):

		for weight in range(0,len(w)):
			accuracy = []
		
			for part in range(0, PARTS):
				train = df[test_idx != part]
				test = df[test_idx == part]

				knn = neighbors.KNeighborsClassifier(n_neighbors = n, weights = w[weight])
				knn.fit(train[features], train[label])
				preds = knn.predict(test[features])

				acc = (np.where(preds== test[label], 1, 0).sum() * 100)/ float(len(test))
				accuracy.append(acc)
		
			overall_accuracy = sum(accuracy) / float(len(accuracy))
			results[names[weight]].append([n, overall_accuracy])
			print "Neighbors: %d, Accuracy: %f, Weights: %s" %(n, overall_accuracy, names[weight])

	plots( results, normalization )

# Plots the results.  Lets matplotlib choose defaults
def plots( results, normalization ):

	result_uniform = pd.DataFrame(results['uniform'], columns = ['n', 'accuracy'])
	result_distance = pd.DataFrame(results['distance'], columns = ['n', 'accuracy'])

	plt.plot(result_uniform.n, result_uniform.accuracy, label = "Uniform")
	plt.plot(result_distance.n, result_distance.accuracy, label = "Distance")
	
	plt.xlabel('Varying Values of K')
	plt.xticks(result_uniform.n)

	plt.ylabel('Accuracy')
	
	title = "Accuracy of Kth Nearest Neighbor on " + INFILE[:-4].upper() + ", " + normalization
	plt.title(title)
	
	plt.legend()
	
	save_file = "accuracy_" + str(INFILE[:-4]) + "_" + normalization + ".png"
	plt.savefig(save_file)
	
	plt.close()


# Changes each of the columns except the label into z-score normalized
# random variables.
def z_score_normalize( df, features ):

	for col in features:
		df[col] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

	return df

main(sys.argv)